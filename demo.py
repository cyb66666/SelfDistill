import torch
from src import open_clip
from src.training.data import create_rs3_dataloader


print("=" * 60)
print("Demo: Loading RS5M ViT-B-32 Model")
print("=" * 60)

# 使用 create_model加载模型
visual_checkpoint = 'checkpoints/best_model.pth'
print(f"\nLoading model from {visual_checkpoint}...")
model = open_clip.create_model(
    'ViT-B-32',
    pretrained=False,
    precision='fp32',
    device='cpu'
).visual

model.load_state_dict(torch.load(visual_checkpoint)['state_dict'])

model.eval()

print(f"\n✓ Model loaded successfully!")

# ==================== 配置参数 ====================
RS3_TAR_DIR = "./rs3"      # RS3 数据集 tar 文件所在目录
BATCH_SIZE = 4             # 每个批次的样本数量
NUM_WORKERS = 2            # 数据加载的并行工作进程数
    
print("正在测试 RS3GridDistillDataset...")
    
# ==================== 创建数据集 ====================
dataloader = create_rs3_dataloader(
        rs3_tar_dir='./rs3_val',
        batch_size=8,
        num_workers=8,
        whole_image_size=1024,
        crop_size=224,
        max_split=4,
        max_boxes=16,
        crop_scale=1.0,
        shuffle=True,
        drop_last=False,
        distributed=False,
        world_size=1,
        rank=0,
        split='val'
    )

for batch_idx, batch_data in enumerate(dataloader):
    # 解包返回的数据
    # images: 整图张量 [batch_size, 3, 1024, 1024]
    # boxes_templates: 边界框模板 [batch_size, max_boxes, 5] (xyxy + 有效标志)
    # image_crops_templates: 裁剪区域张量 [batch_size, max_boxes, 3, 224, 224]
    # masks: 有效区域掩码 [batch_size, max_boxes] (True 表示有效，False 表示填充)
    # img_names: 图片名称列表
    images, boxes_templates, image_crops_templates, masks, img_names = batch_data
    print(images.shape)
    student_features = model._eval(images)
    print(f"Student's feature shape: {student_features.shape}")
    break