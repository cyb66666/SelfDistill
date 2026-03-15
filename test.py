import torch
from src import open_clip
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


# 第一步：加载完整的官方 RS5M 预训练 CLIP 模型（包含文本和视觉编码器）
official_model = open_clip.create_model(
        "ViT-B-32",
        pretrained='pretrained/RS5M_ViT-B-32.pt',
        precision="fp32",
        device='cuda'
    )

# 第二步：创建新的模型实例（不加载预训练权重）
model = open_clip.create_model(
        "ViT-B-32",
        pretrained=None,  # 不加载预训练权重
        precision="fp32",
        device='cuda'
    )

# 第三步：将官方模型的文本编码器权重复制到新模型
text_state_dict = {}
for key in official_model.state_dict().keys():
    if 'text.' in key or 'token_embedding' in key or 'ln_final' in key or 'text_projection' in key:
        text_state_dict[key] = official_model.state_dict()[key]

# 加载官方文本编码器权重
model.load_state_dict(text_state_dict, strict=False)
print(f"✓ 加载官方文本编码器权重:")
print(f"  成功加载的键数：{len(text_state_dict)}")

# 第四步：加载学生模型的视觉编码器权重
print(f"\n✓ 加载学生模型视觉编码器权重...")
visual_state_dict = torch.load("/workspace/SelfDistill/checkpoints/best_model.pth", map_location='cpu')["state_dict"]
model.visual.load_state_dict(visual_state_dict)
print(f"✓ 视觉编码器权重加载完成")

# 释放官方模型内存
del official_model
import gc
gc.collect()

model.eval()

#模型加载到GPU
model.to(device="cuda", dtype=torch.float32)

#加载/workspace/SelfDistill/max_resolution_image.png的图片
image_path = 'image.png'
image = Image.open(image_path).convert('RGB')
mean=[0.3759, 0.3912, 0.3618]  # RS3 数据集的通道均值
std=[0.2582, 0.2472, 0.2461]    # RS3 数据集的通道标准差
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # 调整大小
    transforms.ToTensor(),  # 转为 tensor [0,1]
    transforms.Normalize(  # 标准化
        mean=mean, 
        std=std
    )
])

image_tensor = transform(image).unsqueeze(0)
image_tensor = image_tensor.to(device="cuda", dtype=torch.float32)
# 进行预测
with torch.no_grad():
    pooled, student_features_map = model.visual(image_tensor) # pooled: [B, D], student_features_map: [B, D, H, W]
    print(f"pooled shape: {pooled.shape}, student_features_map shape: {student_features_map.shape}")
    
    # 注意：student_features_map 已经是经过 proj 和归一化的特征，不需要额外处理
    # 如果需要进一步处理，可以在这里添加
    
    # 文本编码（使用官方预训练的文本编码器）
    text_str = "sea"
    text = [text_str]  # 使用列表以支持 batch
    tokens = open_clip.tokenize(text)  # 返回 LongTensor
    tokens = tokens.to(device="cuda")
    text_features = model.encode_text(tokens, normalize=True).to(dtype=torch.float32)  # [B, D]
    print(f"\n✓ 文本特征提取完成：{text_str}, shape: {text_features.shape}")
    #计算文本特征和视觉特征图的相似度热力图
    student_features_map = student_features_map @ model.visual.proj
    # 将特征图展平为 [B, D, H*W]
    student_features_flat = student_features_map.reshape(1, 512, 32 * 32)
    student_features_flat = F.normalize(student_features_flat, dim=1)

    #减去全局偏差pooled
    student_features_flat = student_features_flat - pooled.unsqueeze(2)

    # 计算相似度：text_features [B, D] @ student_features_flat [B, D, HW] -> [B, HW]
    similarity_map = torch.matmul(text_features.unsqueeze(1), student_features_flat).squeeze(1)
    # similarity_map = similarity_map / (similarity_map.norm(dim=-1, keepdim=True) + 1e-8)
    
    # 将 [HW] 重塑为空间网格 [H, W]
    sim_grid = similarity_map[0].reshape(32, 32).cpu().numpy()
    
    # 归一化到 [0, 1] 用于可视化
    sim_grid = (sim_grid - sim_grid.min()) / (sim_grid.max() - sim_grid.min() + 1e-8)
    
    #在原图上绘制相似度热力图
    import cv2
    import numpy as np
    image_cv = cv2.imread(image_path)
    image_cv = cv2.resize(image_cv, (1024, 1024))
    
    # 将 32x32 的热力图插值到 1024x1024
    sim_vis = (sim_grid * 255).astype(np.uint8)
    similarity_map_color = cv2.resize(sim_vis, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    similarity_map_color = cv2.applyColorMap(similarity_map_color, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(image_cv, 0.5, similarity_map_color, 0.5, 0)
    output_filename = f"{text_str}_output.png"
    cv2.imwrite(output_filename, overlay)
    print(f"✓ 热力图已保存：{output_filename}")
