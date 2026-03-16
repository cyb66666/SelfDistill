import torch
from src import open_clip
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2
import numpy as np
import gc

# ==================== 第一步：加载完整的官方RS5M模型 ====================
print("正在加载官方RS5M预训练模型...")
model = open_clip.create_model(
    "ViT-B-32",
    pretrained='pretrained/RS5M_ViT-B-32.pt',
    precision="fp32",
    device='cuda'
)
print("✓ 官方模型加载完成")

# # ==================== 第二步：加载学生模型的视觉编码器权重 ====================
print("\n正在加载学生模型视觉编码器权重...")
student_checkpoint = torch.load("/workspace/SelfDistill/checkpoints/best_model.pth", map_location='cpu')
visual_state_dict = student_checkpoint["state_dict"]

# 关键修正：加载视觉编码器权重
# 注意：有些checkpoint可能包含"visual."前缀，有些可能没有
try:
    # 尝试直接加载
    model.visual.load_state_dict(visual_state_dict, strict=False)
    print("✓ 视觉编码器权重加载完成（直接加载）")
except Exception as e:
    print(f"直接加载失败，尝试处理键名: {e}")
    
    # 如果失败，可能需要处理键名
    new_visual_state_dict = {}
    for k, v in visual_state_dict.items():
        # 移除可能的前缀（如"visual."、"module."等）
        if k.startswith('visual.'):
            new_key = k.replace('visual.', '')
        elif k.startswith('module.'):
            new_key = k.replace('module.', '')
        else:
            new_key = k
        new_visual_state_dict[new_key] = v
    
    model.visual.load_state_dict(new_visual_state_dict, strict=False)
    print("✓ 视觉编码器权重加载完成（键名处理后）")

# ==================== 第三步：设置为评估模式 ====================
model.eval()
model.to(device="cuda", dtype=torch.float32)
print("\n✓ 模型已设置为评估模式并移至GPU")

# ==================== 第四步：图像预处理 ====================
h = 1024
w = 1024
patch = h // 32
image_path = 'test_image/sample_007_millionaid_P0430326.jpg'
text_str = "forest"
print(f"\n正在加载图像: {image_path}")
image = Image.open(image_path).convert('RGB')

# RS3 数据集的归一化参数
mean = [0.3759, 0.3912, 0.3618]
std = [0.2582, 0.2472, 0.2461]

transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

image_tensor = transform(image).unsqueeze(0).to(device="cuda", dtype=torch.float32)
print(f"✓ 图像预处理完成，tensor shape: {image_tensor.shape}")

# ==================== 第五步：进行预测和可视化 ====================
with torch.no_grad():
    # 视觉编码
    pooled, student_features_map = model.visual(image_tensor)
    print(f"pooled shape: {pooled.shape}, student_features_map shape: {student_features_map.shape}")
    
    # 文本编码
    text = [text_str]
    tokens = open_clip.tokenize(text).to(device="cuda")
    text_features = model.encode_text(tokens, normalize=True).to(dtype=torch.float32)
    print(f"✓ 文本特征提取完成：'{text_str}', shape: {text_features.shape}")
    
    # ====== 关键修正：计算相似度热力图 ======
    # 1. 将特征图投影到与文本相同的空间
    if hasattr(model.visual, 'proj') and model.visual.proj is not None:
        # 如果 visual 模块有 proj 层
        student_features_map = student_features_map @ model.visual.proj
    else:
        # 如果没有独立的 proj 层，可能需要使用模型的 text_projection
        print("警告：model.visual 没有 proj 属性，使用 model.text_projection")
        student_features_map = student_features_map @ model.text_projection
    
    # 2. 重塑特征图 [B, D, H, W] -> [B, D, H*W]
    student_features_flat = student_features_map.reshape(1, 512, patch * patch)
    student_features_flat = F.normalize(student_features_flat, dim=1)
    
    # 3. 减去全局特征（可选，根据你的需求）
    # pooled_norm = F.normalize(pooled, dim=1).unsqueeze(2)  # [B, D, 1]
    # student_features_flat = student_features_flat - pooled_norm
    
    # 4. 计算相似度 [B, D] @ [B, D, HW] -> [B, HW]
    similarity_map = torch.matmul(text_features.unsqueeze(1), student_features_flat).squeeze(1)
    
    # 5. 重塑为空间网格 [B, H*W] -> [B, 1, H, W]
    sim_grid = similarity_map[0].reshape(1, 1, patch, patch)
    
    # 6. 使用双线性插值上采样到原图大小
    logit_size = (h, w)
    sim_grid_interpolated = F.interpolate(sim_grid, size=logit_size, mode='bilinear', align_corners=False)
    sim_grid = sim_grid_interpolated[0, 0].cpu().numpy()
    
    # 7. 归一化
    sim_grid = (sim_grid - sim_grid.min()) / (sim_grid.max() - sim_grid.min() + 1e-8)
    
    # ====== 可视化 ======
    image_cv = cv2.imread(image_path)
    image_cv = cv2.resize(image_cv, (h, w))
    
    # 将热力图转换为 uint8 并应用颜色映射
    sim_vis = (sim_grid * 255).astype(np.uint8)
    similarity_map_color = cv2.applyColorMap(sim_vis, cv2.COLORMAP_JET)
    
    # 叠加原图和热力图
    overlay = cv2.addWeighted(image_cv, 0.5, similarity_map_color, 0.5, 0)
    
    output_filename = f"output/{text_str}_output_{h}.png"
    cv2.imwrite(output_filename, overlay)
    print(f"✓ 热力图已保存：{output_filename}")

# 可选：释放GPU内存
torch.cuda.empty_cache()
gc.collect()
print("\n✓ 处理完成！")