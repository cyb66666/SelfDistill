# SelfDistill - RS3 知识蒸馏训练项目

基于 OpenCLIP ViT 的知识蒸馏训练框架，使用 Teacher-Student 架构在 RS3 遥感数据集上进行自蒸馏训练。

## 📁 项目结构

```
SelfDistill/
├── README.md                    # 项目说明文档
├── demo.py                      # 快速演示脚本
├── .gitignore                   # Git 忽略文件配置
│
├── scripts/                     # 训练启动脚本
│   └── run_train.sh            # 训练脚本（支持单卡/多卡）
│
├── src/                         # 核心源代码
│   ├── open_clip/              # OpenCLIP 模型实现
│   │   ├── model.py            # Vision Transformer 模型
│   │   ├── transformer.py      # Transformer 层实现
│   │   ├── factory.py          # 模型工厂函数
│   │   └── ...                 # 其他 OpenCLIP 组件
│   └── training/               # 训练相关代码
│       ├── train_distill.py    # 主训练脚本
│       ├── data.py             # 数据加载器 (RS3GridDistillDataset)
│       └── scheduler.py        # 学习率调度器
│
├── pretrained/                  # 预训练权重目录
│   └── RS5M_ViT-B-32.pt        # 教师模型预训练权重
│
├── rs3/                         # RS3 训练数据集
│   ├── rs3-1024-000000.tar
│   ├── rs3-1024-000001.tar
│   └── ...
│
├── rs3_val/                     # RS3 验证数据集
│   └── rs3-1024-000030.tar
│
├── checkpoints/                 # 模型保存目录（运行时生成）
│   ├── best_model.pth          # 最佳验证集模型
│   ├── checkpoint_epoch_*.pth  # 各 epoch 的 checkpoint
│   └── train_*.log             # 训练日志
│
├── logs/                        # 日志目录（运行时生成）
│   └── train_*.log             # 详细训练日志
│
└── docs/                        # 文档目录（可选）
    ├── TRAINING_README.md      # 训练指南
    └── ...                     # 其他文档
```

## 🚀 快速开始

### 1. 环境要求

```bash
# Python 3.10+
# PyTorch 2.1.0+ with CUDA 12.1
# 其他依赖见 requirements.txt
```

### 2. 测试环境

```bash
cd /workspace/SelfDistill
python demo.py
```

### 3. 开始训练

**单卡训练：**
```bash
bash scripts/run_train.sh
```

**多卡训练：**
编辑 `scripts/run_train.sh`，设置 `USE_SINGLE_GPU=false`，然后运行：
```bash
bash scripts/run_train.sh
```

### 4. 查看训练进度

```bash
# 实时查看日志
tail -f logs/train_*.log

# 查看进度条（训练时自动显示）
# 包含：loss、学习率、ETA、剩余时间等
```

## 📊 训练配置

主要配置在 `scripts/run_train.sh` 中：

```bash
# 数据配置
BATCH_SIZE=4
NUM_WORKERS=2
VAL_TAR_COUNT=2          # 验证集使用 2 个 tar 文件（6.25%）

# 训练配置
EPOCHS=6
LR=1e-5
SCHEDULER_TYPE="cosine"  # 学习率调度器

# 损失函数
LOSS_TYPE="cosine_sim"   # 余弦相似度损失
```

## 🎯 核心功能

### 1. 双模型知识蒸馏
- **Teacher 模型**：预训练 ViT-B-32，参数冻结
- **Student 模型**：复制 teacher 初始权重，可训练
- **目标**：Student 学习 Teacher 的特征表示

### 2. 数据分割
- **训练集**：前 30 个 tar 文件（93.75%）
- **验证集**：最后 2 个 tar 文件（6.25%）

### 3. 训练特性
- ✅ 自动混合精度（AMP）
- ✅ 学习率预热 + 余弦退火
- ✅ 梯度裁剪
- ✅ 多 GPU 分布式训练（DDP）
- ✅ Checkpoint 保存与恢复
- ✅ 实时进度条和 ETA 预估

## 📈 训练输出

训练完成后，在 `checkpoints/` 目录查看：

```
checkpoints/
├── best_model.pth              # 最佳验证集模型
├── checkpoint_epoch_1.pth      # Epoch 1 的 checkpoint
├── checkpoint_epoch_2.pth      # Epoch 2 的 checkpoint
└── train_20260309_120000.log  # 训练日志
```

## 🔧 故障排查

### 问题 1: CUDA 内存不足
```bash
# 减小 batch size
BATCH_SIZE=2

# 或使用更少的 workers
NUM_WORKERS=1
```

### 问题 2: 损失不下降
```bash
# 检查学习率是否合适
LR=2e-5  # 尝试更小的学习率

# 或增加 warmup
WARMUP_STEPS=2000
```

### 问题 3: 数据加载慢
```bash
# 增加 workers 数量
NUM_WORKERS=4

# 或检查磁盘 IO
```

## 📚 详细文档

- **[训练指南](docs/TRAINING_README.md)** - 完整的训练参数说明
- **[快速开始](docs/快速开始-Cosine_Loss.md)** - Cosine 损失快速上手
- **[损失函数说明](docs/余弦相似度损失说明.md)** - 损失函数公式和实现细节

## 📝 命令行参数

完整参数列表：

```bash
python src/training/train_distill.py --help
```

常用参数：
- `--rs3-tar-dir`: RS3 数据集路径
- `--batch-size`: 批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--loss-type`: 损失类型（mse/l1/cosine/cosine_sim）
- `--scheduler-type`: 调度器类型（cosine/constant/cooldown）
- `--output-dir`: 输出目录（默认 ./checkpoints）
- `--resume`: 恢复训练的 checkpoint 路径

## 💡 使用示例

### 自定义配置训练

```bash
python src/training/train_distill.py \
    --rs3-tar-dir ./rs3 \
    --batch-size 8 \
    --epochs 10 \
    --lr 2e-5 \
    --loss-type cosine_sim \
    --scheduler-type cosine \
    --warmup-steps 1000 \
    --output-dir ./my_checkpoints
```

### 从 checkpoint 恢复

```bash
python src/training/train_distill.py \
    --rs3-tar-dir ./rs3 \
    --resume ./checkpoints/checkpoint_epoch_3.pth
```

## 🎓 架构说明

### Teacher-Student 架构

```
整图 [B, 3, 1024, 1024]
  ↓
Student 编码（可训练）
  ↓
student_features: [B*16, dim]

裁剪区域 [B, 16, 3, 224, 224]
  ↓
Teacher 编码（冻结）
  ↓
teacher_features: [B*16, dim]

↓ 计算损失
L = mean(1 - cos(student_features, teacher_features))
```

### 关键设计

1. **独立模型**：Teacher 和 Student 是两个独立的模型实例
2. **相同初始化**：Student 复制 Teacher 的初始权重
3. **梯度控制**：
   - Teacher: `requires_grad=False` + `eval()`
   - Student: `requires_grad=True` + `train()`
4. **优化器**：只优化 Student 的参数

## 🤝 贡献

如有问题或建议，请提 issue 或 PR。

## 📄 许可证

本项目遵循原 OpenCLIP 和 RS3 数据集的许可协议。
