"""
RS3 网格蒸馏训练脚本

功能：
1. 从 RS3 数据集加载遥感图像
2. 使用预训练的 ViT-B-32 作为 teacher
3. 训练 student 模型进行知识蒸馏
4. 支持混合精度训练、checkpoint 保存等功能
"""

import os
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src import open_clip
from src.training.data import create_rs3_dataloader
from tools.scheduler import cosine_lr, const_lr, const_lr_cooldown


# ==================== 配置参数 ====================
class TrainConfig:
    """训练配置类"""
    
    # ============ 数据配置 ============
    rs3_tar_dir: str = "./rs3"           # RS3 数据集目录
    rs3_val_dir: str = "./rs3_val"
    batch_size: int = 4                   # 批次大小
    num_workers: int = 2                  # 数据加载进程数
    whole_image_size: int = 1024          # 整图尺寸
    crop_size: int = 224                  # Crop 尺寸
    max_split: int = 4                    # 最大网格划分 (4×4=16)
    max_boxes: int = 16                   # 最大 box 数量
    crop_scale: float = 1.0               # Crop 缩放比例
    val_tar_count: int = 2                # 验证集 tar 文件数量（默认 2 个，约 6.25%）
    
    # ============ 模型配置 ============
    model_name: str = 'ViT-B-32'          # 模型名称
    pretrained_path: str = "pretrained/RS5M_ViT-B-32.pt"  # 预训练权重路径
    precision: str = 'fp32'               # 精度设置 (fp32/fp16/bf16)
    
    # ============ 训练配置 ============
    epochs: int = 10                      # 训练轮数
    lr: float = 5e-5                      # 基础学习率
    weight_decay: float = 0.05            # 权重衰减
    warmup_epochs: int = 1                # 预热轮数（基于 epoch）
    warmup_steps: int = 1000              # 预热步数（基于 step，优先级更高）
    grad_clip: float = 1.0                # 梯度裁剪阈值
    optimizer_betas: tuple = (0.9, 0.98)  # AdamW 动量参数
    optimizer_eps: float = 1e-8           # AdamW 数值稳定性参数
    scheduler_type: str = 'cosine'        # 学习率调度器类型 ('cosine', 'constant', 'cooldown')
    cooldown_steps: int = 0               # 冷却阶段步数（仅当 scheduler_type='cooldown' 时有效）
    
    # ============ 输出配置 ============
    output_dir: str = "./checkpoints"     # 模型 checkpoint 保存目录
    log_dir: str = "./logs"               # 日志文件保存目录
    save_freq: int = 1                    # checkpoint 保存频率 (每 N 个 epoch 保存一次)
    log_freq: int = 10                    # 日志打印频率 (每 N 个 batch 打印一次)
    
    # ============ 设备配置 ============
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp: bool = True                  # 是否使用自动混合精度
    
    # ============ 分布式训练配置 ============
    distributed: bool = False             # 是否启用分布式训练
    rank: int = 0                         # 当前进程排名
    world_size: int = 1                   # 总进程数
    local_rank: int = 0                   # 本地 GPU 排名


def setup_logging(output_dir: str, rank: int = 0):
    """设置日志输出"""
    log_file = os.path.join(output_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}_rank{rank}.log')
    
    # 只在 rank 0 打印日志到控制台
    handlers = [logging.FileHandler(log_file)]
    if rank == 0:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - [Rank {rank}] - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def build_models(config: TrainConfig, logger: logging.Logger):
    """
    构建 Teacher 和 Student 两个独立模型
    
    Args:
        config: 训练配置
        logger: 日志记录器
    
    Returns:
        teacher_model: 冻结的 teacher 模型
        student_model: 可训练的 student 模型
    """
    logger.info(f"Loading models from {config.pretrained_path}...")
    
    # 1. 加载 Teacher 模型（预训练权重）
    logger.info("\n=== Loading Teacher Model ===")
    teacher_model = open_clip.create_model(
        config.model_name,
        pretrained=config.pretrained_path,
        precision=config.precision,
        device='cpu'
    ).visual
    
    # Teacher 模型冻结所有参数
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    logger.info(f"✓ Teacher model loaded and frozen")
    logger.info(f"  - Model type: {type(teacher_model).__name__}")
    logger.info(f"  - Parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    logger.info(f"  - Requires grad: {sum(p.requires_grad for p in teacher_model.parameters()):,} (frozen)")
    
    # 2. 加载 Student 模型（复制 teacher 的初始权重）
    logger.info("\n=== Loading Student Model ===")
    student_model = open_clip.create_model(
        config.model_name,
        pretrained=config.pretrained_path,
        precision=config.precision,
        device='cpu'
    ).visual
    
    # Student 模型保持可训练状态
    student_model.train()
    
    logger.info(f"✓ Student model loaded with same initial weights")
    logger.info(f"  - Model type: {type(student_model).__name__}")
    logger.info(f"  - Parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    logger.info(f"  - Requires grad: {sum(p.requires_grad for p in student_model.parameters()):,} (trainable)")
    
    return teacher_model, student_model


def build_dataloader(config: TrainConfig, split: str = 'train', shuffle: bool = True):
    """
    构建数据加载器
    
    Args:
        config: 训练配置
        split: 数据集分割 ('train' 或 'val')
        shuffle: 是否打乱数据
    
    Returns:
        dataloader: 数据加载器
    """
    if split == 'train':
        dir = config.rs3_tar_dir
    else:
        dir = config.rs3_val_dir

    dataloader = create_rs3_dataloader(
        rs3_tar_dir=dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        whole_image_size=config.whole_image_size,
        crop_size=config.crop_size,
        max_split=config.max_split,
        max_boxes=config.max_boxes,
        crop_scale=config.crop_scale,
        shuffle=shuffle and not config.distributed,  # 分布式时使用 DistributedSampler
        drop_last=False,
        distributed=config.distributed,
        world_size=config.world_size,
        rank=config.rank,
        split=split
    )
    
    return dataloader


class DistillLoss(nn.Module):
    """
    知识蒸馏损失
    
    支持多种损失类型：
    1. MSE Loss: 均方误差
    2. L1 Loss: 平均绝对误差
    3. Cosine Embedding Loss: 余弦嵌入损失
    4. Cosine Similarity Loss: 逐位置余弦相似度损失（公式：L = 1/(m*n) * ΣΣ(1 - cos(s_ij, t_ij))）
    """
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'cosine':
            self.criterion = nn.CosineEmbeddingLoss(reduction='none')
        elif loss_type == 'cosine_sim':
            # 逐位置余弦相似度损失
            pass  # 不需要额外的 criterion
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, student_features: torch.Tensor, 
                teacher_features: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        计算蒸馏损失
        
        Args:
            student_features: Student 特征 [N, dim]
            teacher_features: Teacher 特征 [N, dim]
            mask: 有效区域掩码 [N] (True=有效)
        
        Returns:
            loss: 标量损失值
        """
        # 只计算有效区域的损失
        valid_mask = mask.view(-1)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=student_features.device)
        
        student_valid = student_features[valid_mask]
        teacher_valid = teacher_features[valid_mask]
        
        if self.loss_type in ['mse', 'l1']:
            loss = self.criterion(student_valid, teacher_valid).mean(dim=-1)
            loss = loss.mean()
        elif self.loss_type == 'cosine':
            target = torch.ones(student_valid.shape[0], device=student_valid.device)
            loss = self.criterion(student_valid, teacher_valid, target).mean()
        elif self.loss_type == 'cosine_sim':
            # 逐位置余弦相似度损失
            # L = 1/(m*n) * ΣΣ(1 - cos(s_ij, t_ij))
            # cos(a,b) = (a·b) / (|a| * |b|)
            
            # 归一化特征向量
            student_norm = F.normalize(student_valid, p=2, dim=-1)  # [N_valid, dim]
            teacher_norm = F.normalize(teacher_valid, p=2, dim=-1)  # [N_valid, dim]
            
            # 计算逐位置余弦相似度
            cosine_sim = (student_norm * teacher_norm).sum(dim=-1)  # [N_valid]
            
            # 损失 = 1 - 余弦相似度
            loss = (1 - cosine_sim).mean()
        
        return loss


def _get_warmup_lr(optimizer, step: int, warmup_length: int, base_lr: float):
    """
    计算预热阶段的学习率（已废弃，使用 scheduler.py 中的函数）
    
    Args:
        optimizer: 优化器
        step: 当前步数（从 0 开始）
        warmup_length: 预热总步数
        base_lr: 基础学习率
    
    Returns:
        lr: 当前学习率
    """
    # lr = base_lr × (step + 1) / warmup_length
    lr = base_lr * float(step + 1) / warmup_length
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def train_one_epoch(
    teacher_model: nn.Module,
    student_model: nn.Module,
    dataloader: DataLoader,
    criterion: DistillLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    config: TrainConfig,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    训练一个 epoch
    
    Args:
        teacher_model: 冻结的 teacher 模型
        student_model: 可训练的 student 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scaler: AMP 梯度缩放器
        epoch: 当前 epoch
        config: 配置
        logger: 日志记录器
    
    Returns:
        metrics: 训练指标字典
    """
    student_model.train()
    teacher_model.eval()  # 确保 teacher 保持评估模式
    
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    global_step = epoch * len(dataloader)  # 计算全局步数（用于预热）
    
    # 只在 rank 0 显示进度条
    use_tqdm = not config.distributed or config.rank == 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.epochs}', disable=not use_tqdm)
    
    for batch_idx, batch_data in enumerate(pbar):
        current_step = global_step + batch_idx  # 当前全局步数
        
        # 使用 scheduler 更新学习率（替代原来的 _get_warmup_lr）
        current_lr = scheduler(current_step)
        
        # 解包数据
        images, boxes_templates, image_crops_templates, masks, img_names = batch_data
        
        # 移动到设备
        images = images.to(config.device, non_blocking=True)
        boxes_templates = boxes_templates.to(config.device, non_blocking=True)
        image_crops_templates = image_crops_templates.to(config.device, non_blocking=True)
        masks = masks.to(config.device, non_blocking=True)
        
        # 前向传播（使用 AMP）
        with autocast(enabled=config.use_amp):
            # Teacher 编码（不计算梯度，使用独立的 teacher 模型）
            with torch.no_grad():
                teacher_features = teacher_model.teacher_roi_encode(image_crops_templates, masks)
            
            # Student 编码（使用独立的 student 模型）
            # 如果是 DDP 包装的模型，需要通过 .module 访问原始模型
            if hasattr(student_model, 'module'):
               student_features = student_model.module.student_encode(images, boxes_templates, teacher_features)
            else:
               student_features = student_model.student_encode(images, boxes_templates, teacher_features)
            
            # 计算损失
            loss = criterion(student_features, teacher_features, masks.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # 梯度裁剪（只对 student 模型）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.grad_clip)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        if use_tqdm:
            avg_loss = total_loss / num_batches
            elapsed_time = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed_time if elapsed_time > 0 else 0
            remaining_batches = len(dataloader) - (batch_idx + 1)
            eta_seconds = remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            postfix_dict = {
                'loss': f'{avg_loss:.4f}',
                'time': f'{elapsed_time/60:.1f}m',
                'eta': f'{eta_seconds/60:.1f}m',
                'lr': f'{current_lr:.2e}'
            }
            
            # 显示预热状态
            if config.warmup_steps > 0 and current_step < config.warmup_steps:
                postfix_dict['warmup'] = f'{current_step}/{config.warmup_steps}'
            
            pbar.set_postfix(postfix_dict)
        
        # 打印日志（保持原有逻辑）
        if (batch_idx + 1) % config.log_freq == 0 and use_tqdm:
            avg_loss = total_loss / num_batches
            elapsed_time = time.time() - start_time
            logger.info(
                f"Epoch [{epoch+1}/{config.epochs}] | "
                f"Batch [{batch_idx+1}/{len(dataloader)}] | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {elapsed_time:.2f}s"
            )
    
    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'time': time.time() - start_time
    }


@torch.no_grad()
def validate(
    teacher_model: nn.Module,
    student_model: nn.Module,
    dataloader: DataLoader,
    criterion: DistillLoss,
    epoch: int,
    config: TrainConfig,
    logger: logging.Logger
) -> Dict[str, float]:

    student_model.eval()
    teacher_model.eval()

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    # 只在主进程显示进度条
    show_bar = (not config.distributed) or config.rank == 0

    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Validate Epoch {epoch+1}",
        disable=not show_bar
    )

    for batch_idx, batch_data in enumerate(pbar):
        images, boxes_templates, image_crops_templates, masks, img_names = batch_data

        images = images.to(config.device, non_blocking=True)
        boxes_templates = boxes_templates.to(config.device, non_blocking=True)
        image_crops_templates = image_crops_templates.to(config.device, non_blocking=True)
        masks = masks.to(config.device, non_blocking=True)

        # forward
        teacher_features = teacher_model.teacher_roi_encode(image_crops_templates, masks)

        if hasattr(student_model, 'module'):
            student_features = student_model.module.student_encode(
                images, boxes_templates, teacher_features
            )
        else:
            student_features = student_model.student_encode(
                images, boxes_templates, teacher_features
            )

        loss = criterion(student_features, teacher_features, masks.view(-1))

        total_loss += loss.item()
        num_batches += 1

        # 更新进度条信息
        if show_bar:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    logger.info(
        f"Validation | Epoch [{epoch+1}/{config.epochs}] | "
        f"Val Loss: {avg_loss:.4f} | "
        f"Time: {time.time() - start_time:.2f}s"
    )

    return {
        "val_loss": avg_loss,
        "val_time": time.time() - start_time
    }


def save_checkpoint(
    student_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    output_dir: str,
    filename: str = 'checkpoint.pth'
):
    """
    保存 checkpoint（只保存 student 模型）
    
    Args:
        student_model: 可训练的 student 模型
        optimizer: 优化器
        epoch: 当前 epoch
        loss: 损失值
        output_dir: 输出目录
        filename: 文件名
    """
    checkpoint_path = os.path.join(output_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'state_dict': student_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    student_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    logger: logging.Logger
) -> int:
    """
    加载 checkpoint（只恢复 student 模型）
    
    Args:
        student_model: 可训练的 student 模型
        optimizer: 优化器
        checkpoint_path: checkpoint 路径
        logger: 日志记录器
    
    Returns:
        start_epoch: 起始 epoch
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    student_model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    logger.info(f"✓ Checkpoint loaded (epoch {start_epoch-1}, loss: {checkpoint.get('loss', 'N/A')})")
    
    return start_epoch


def main():
    """主训练函数"""
    
    # ==================== 解析命令行参数 ====================
    parser = argparse.ArgumentParser(description='RS3 Grid Distillation Training')
    
    # 数据配置
    parser.add_argument('--rs3-tar-dir', type=str, default='./rs3',
                        help='RS3 train 文件目录')
    parser.add_argument('--rs3-val-dir', type=str, default='./rs3_val',
                        help='RS3 val 文件目录')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批次大小 (default: 4)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='数据加载进程数 (default: 2)')
    
    # 模型配置
    parser.add_argument('--model-name', type=str, default='ViT-B-32',
                        help='模型名称 (default: ViT-B-32)')
    parser.add_argument('--pretrained-path', type=str, default='pretrained/RS5M_ViT-B-32.pt',
                        help='预训练权重路径')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='精度设置 (default: fp32)')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数 (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='基础学习率 (default: 5e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='权重衰减 (default: 0.05)')
    parser.add_argument('--warmup-epochs', type=int, default=1,
                        help='预热轮数（基于 epoch，默认：1）')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='预热步数（基于 step，优先级更高，默认：1000）')
    parser.add_argument('--scheduler-type', type=str, default='cosine',
                        choices=['cosine', 'constant', 'cooldown'],
                        help='学习率调度器类型 (default: cosine)')
    parser.add_argument('--cooldown-steps', type=int, default=0,
                        help='冷却阶段步数（仅当 scheduler_type=cooldown 时有效）')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='梯度裁剪阈值 (default: 1.0)')
    parser.add_argument('--loss-type', type=str, default='mse',
                        choices=['mse', 'l1', 'cosine', 'cosine_sim'],
                        help='损失函数类型 (default: mse)\n'
                             '  mse: 均方误差\n'
                             '  l1: 平均绝对误差\n'
                             '  cosine: 余弦嵌入损失\n'
                             '  cosine_sim: 逐位置余弦相似度损失（推荐用于密集特征蒸馏）')
    parser.add_argument('--optimizer-betas', type=tuple, default=(0.9, 0.98),
                        help='AdamW 动量参数 (default: (0.9, 0.98))')
    parser.add_argument('--optimizer-eps', type=float, default=1e-8,
                        help='AdamW 数值稳定性参数 (default: 1e-8)')
    parser.add_argument('--val-tar-count', type=int, default=2,
                        help='验证集使用的 tar 文件数量 (default: 2, 约 6.25%)')
    
    # 输出配置
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='模型保存目录 (default: ./checkpoints)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='日志文件保存目录 (default: ./logs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的 checkpoint 路径')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='checkpoint 保存频率 (default: 1)')
    parser.add_argument('--log-freq', type=int, default=10,
                        help='日志打印频率 (default: 10)')
    
    # 分布式训练配置
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='本地 GPU 排名（自动设置）')
    parser.add_argument('--world-size', type=int, default=1,
                        help='总进程数（用于分布式训练）')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='分布式后端 (default: nccl)')
    parser.add_argument('--dist-url', type=str, default='env://',
                        help='分布式通信 URL (default: env://)')
    
    # 设备配置
    parser.add_argument('--no-amp', action='store_true',
                        help='禁用自动混合精度')
    
    args = parser.parse_args()
    
        # ==================== 初始化配置 ====================
    config = TrainConfig()
    
    # ==================== 初始化分布式训练 ====================
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 通过环境变量启动（torch.distributed.launch 或 torchrun）
        config.distributed = True
        config.rank = int(os.environ['RANK'])
        config.world_size = int(os.environ['WORLD_SIZE'])
        config.local_rank= int(os.environ.get('LOCAL_RANK', 0))
    elif 'LOCAL_RANK' in os.environ:
        # torchrun 可能只设置了 LOCAL_RANK
        config.distributed = True
        config.local_rank = int(os.environ['LOCAL_RANK'])
        config.rank = config.local_rank
        config.world_size = int(os.environ.get('WORLD_SIZE', 1))
    elif args.local_rank != -1:
        # 手动指定 local_rank
        config.distributed = True
        config.local_rank = args.local_rank
        config.rank = args.local_rank
        config.world_size = args.world_size
    
    # 设置分布式环境
    if config.distributed:
        # 验证配置是否有效
        if config.world_size < 1:
            logging.warning(f"Invalid world_size: {config.world_size}, disabling distributed training")
            config.distributed = False
        elif config.rank >= config.world_size:
            logging.warning(f"Invalid rank {config.rank} for world_size {config.world_size}, disabling distributed training")
            config.distributed = False
        else:
            torch.cuda.set_device(config.local_rank)
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=config.world_size,
                rank=config.rank
            )
        config.device = f'cuda:{config.local_rank}'

    if not config.distributed:
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 用命令行参数覆盖配置
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # 特殊处理
    config.use_amp = not args.no_amp
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ==================== 创建输出目录 ====================
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # ==================== 设置日志 ====================
    logger = setup_logging(config.log_dir, rank=config.rank)
    logger.info("=" * 60)
    logger.info("RS3 Grid Distillation Training")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # ==================== 构建模型 ====================
    logger.info("Building models...")
    teacher_model, student_model = build_models(config, logger)
    
    # 移动模型到设备
    teacher_model = teacher_model.to(config.device)
    student_model = student_model.to(config.device)
    logger.info(f"Models moved to device: {config.device}")
    
    # 分布式训练：只包装 student 模型
    if config.distributed:
        # 验证分布式配置是否有效
        if config.world_size <= 1 or config.rank >= config.world_size:
            logger.warning(f"Invalid distributed configuration: world_size={config.world_size}, rank={config.rank}")
            logger.warning("Falling back to single GPU training")
            config.distributed = False
        else:
            student_model = DDP(student_model, device_ids=[config.local_rank], find_unused_parameters=False)
            logger.info("Student model wrapped with DDP")
    
    # ==================== 构建数据加载器 ====================
    logger.info("\nBuilding dataloaders...")
    train_loader = build_dataloader(config, split='train', shuffle=True)
    val_loader = build_dataloader(config, split='val', shuffle=False)
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    
    # ==================== 构建优化器 ====================
    logger.info("\nBuilding optimizer...")
    
    # 注意：这是两个独立模型的知识蒸馏
    # - Teacher 模型：预训练权重，完全冻结
    # - Student 模型：复制 teacher 初始权重，可训练
    # 优化器只优化 student 模型的参数
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.weight_decay
    )
    logger.info(f"Optimizer: AdamW (lr={config.lr}, betas={config.optimizer_betas}, "
                f"eps={config.optimizer_eps}, weight_decay={config.weight_decay})")
    logger.info(f"Optimizing student model parameters only")
    
    # ==================== 构建学习率调度器 ====================
    total_steps = config.epochs * len(train_loader)
    warmup_steps_for_scheduler = config.warmup_steps if config.warmup_steps > 0 else config.warmup_epochs * len(train_loader)
    
    if config.scheduler_type == 'cosine':
        scheduler = cosine_lr(optimizer, config.lr, warmup_steps_for_scheduler, total_steps)
        logger.info(f"Scheduler: Cosine (warmup_steps={warmup_steps_for_scheduler}, total_steps={total_steps})")
    elif config.scheduler_type == 'constant':
        scheduler = const_lr(optimizer, config.lr, warmup_steps_for_scheduler, total_steps)
        logger.info(f"Scheduler: Constant (warmup_steps={warmup_steps_for_scheduler}, total_steps={total_steps})")
    elif config.scheduler_type == 'cooldown':
        scheduler = const_lr_cooldown(
            optimizer, config.lr, warmup_steps_for_scheduler, total_steps,
            config.cooldown_steps
        )
        logger.info(f"Scheduler: Cooldown (warmup_steps={warmup_steps_for_scheduler}, "
                    f"cooldown_steps={config.cooldown_steps}, total_steps={total_steps})")
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
    # ==================== 构建损失函数 ====================
    criterion = DistillLoss(loss_type=args.loss_type)
    logger.info(f"Loss function: {args.loss_type}")
    
    # ==================== 构建 AMP Scaler ====================
    scaler = GradScaler(enabled=config.use_amp)
    logger.info(f"AMP enabled: {config.use_amp}")
    
    # ==================== 恢复训练（如果有 checkpoint） ====================
    start_epoch = 0
    if args.resume:
        # 分布式训练时只从 rank 0 加载
        if not config.distributed or config.rank == 0:
            start_epoch = load_checkpoint(student_model.module if config.distributed else student_model, optimizer, args.resume, logger)
    
    # ==================== 开始训练 ====================
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\nEpoch [{epoch+1}/{config.epochs}]")
        logger.info("-" * 60)
        
        # 训练
        train_metrics = train_one_epoch(
            teacher_model=teacher_model,
            student_model=student_model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            config=config,
            logger=logger
        )
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Time: {train_metrics['time']:.2f}s")
        
        # 验证
        if val_loader is not None:
            val_metrics = validate(
                teacher_model=teacher_model,
                student_model=student_model,
                dataloader=val_loader,
                criterion=criterion,
                epoch=epoch,
                config=config,
                logger=logger
            )
            # 保存最佳模型
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                # 分布式训练时只从 rank 0 保存
                if not config.distributed or config.rank == 0:
                    save_checkpoint(
                        student_model.module if config.distributed else student_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=val_metrics['val_loss'],
                        output_dir=config.output_dir,
                        filename='best_model.pth'
                    )
                    logger.info(f"✓ New best model saved! (val_loss: {best_val_loss:.4f})")
        
        # 定期保存 checkpoint
        if (epoch + 1) % config.save_freq == 0:
            # 分布式训练时只从 rank 0 保存
            if not config.distributed or config.rank == 0:
                save_checkpoint(
                    student_model.module if config.distributed else student_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=train_metrics['loss'],
                    output_dir=config.output_dir,
                    filename=f'checkpoint_epoch_{epoch+1}.pth'
                )
    
    # ==================== 训练完成 ====================
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 60)
    
    # 清理分布式环境
    if config.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
