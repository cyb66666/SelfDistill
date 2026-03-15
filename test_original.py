import torch
from src import open_clip
from PIL import Image
from torchvision import transforms

#加载pretrained/RS5M_ViT-B-32.pt CLIP
model = open_clip.create_model(
        "ViT-B-32",
        pretrained='pretrained/RS5M_ViT-B-32.pt',
        precision="fp32",
        device='cuda'
    )
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
    # 使用 visual._eval 获取空间特征图 [B, D, H, W] 而不是全局特征
    pooled, student_features_map = model.visual.forward(image_tensor)
    print(pooled.shape, student_features_map.shape)
    student_features_map = student_features_map @ model.visual.proj
    
    student_features_map = student_features_map.reshape(1, 32, 32, 512).permute(0, 3, 1, 2)
    print(f"student_features_map shape: {student_features_map.shape}")
    # 投影到文本空间


    # 文本编码（修正：先 tokenize）
    category = "sea"
    text = [category]  # 使用列表以支持 batch
    tokens = open_clip.tokenize(text)  # 返回 LongTensor
    tokens = tokens.to(device="cuda")
    text_features = model.encode_text(tokens).to(dtype=torch.float32)  # [B, D]
    #计算文本特征和视觉特征图的相似度热力图
    B, D, H, W = student_features_map.shape
    # 将特征图展平为 [B, D, H*W]
    student_features_flat = student_features_map.reshape(B, D, H * W)
    # student_features_flat = student_features_flat - pooled.unsqueeze(2)
    # 计算相似度：text_features [B, D] @ student_features_flat [B, D, HW] -> [B, HW]
    similarity_map = torch.matmul(text_features.unsqueeze(1), student_features_flat).squeeze(1)
    similarity_map = similarity_map / (similarity_map.norm(dim=-1, keepdim=True) + 1e-8)
    
    # 将 [HW] 重塑为空间网格 [H, W]
    sim_grid = similarity_map[0].reshape(H, W).cpu().numpy()
    
    # 归一化到 [0, 1] 用于可视化
    sim_grid = (sim_grid - sim_grid.min()) / (sim_grid.max() - sim_grid.min() + 1e-8)
    
    #在原图上绘制相似度热力图
    import cv2
    import numpy as np
    image_cv = cv2.imread(image_path)
    image_cv = cv2.resize(image_cv, (1024, 1024))
    
    # 将 HxW 的热力图插值到图片大小
    sim_vis = (sim_grid * 255).astype(np.uint8)
    similarity_map_color = cv2.resize(sim_vis, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    similarity_map_color = cv2.applyColorMap(similarity_map_color, cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(image_cv, 0.5, similarity_map_color, 0.5, 0)
    cv2.imwrite(f"{category}_output_teacher.png", overlay)
