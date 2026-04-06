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
import math
import logging
import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Any

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
from src.training.data import create_mgrs_dataloader, create_rs3_dataloader
from src.training.scheduler import cosine_lr, const_lr, const_lr_cooldown

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc, assignment]


# ==================== 配置参数 ====================
class TrainConfig:
    """训练配置类"""
    
    # ============ 数据配置 ============
    dataset: str = "rs3"                 # "rs3"（WebDataset .tar）| "mgrs"（text_info.json + global_imgs）
    mgrs_json_path: str = ""             # MGRS：text_info.json 路径
    mgrs_image_root: str = ""            # MGRS：global_imgs 根目录（与 JSON 中 global_filepath 拼接）
    rs3_tar_dir: str = "./rs3"           # RS3 数据集目录
    rs3_val_dir: str = "./rs3_val"
    batch_size: int = 4                   # 批次大小
    num_workers: int = 2                  # 数据加载进程数
    whole_image_size: int = 1024          # 整图尺寸
    crop_size: int = 224                  # Crop 尺寸
    max_split: int = 7                    # 最大网格划分 (4×4=16)
    max_boxes: int = 49                   # 最大 box 数量
    crop_scale: float = 1.0               # Crop 缩放比例
    val_tar_count: int = 2                # 验证集 tar 文件数量（默认 2 个，约 6.25%）
    train_tar_count: int = 30             # 训练集 tar 文件数量（按排序取前 N 个）
    
    # ============ 模型配置 ============
    model_name: str = 'ViT-B-32'          # 模型名称
    pretrained_path: str = "pretrained/RS5M_ViT-B-32.pt"  # 预训练权重路径
    base_model_type: str = "remoteclip"  # 基础模型类型（remoteclip/georsclip/clip/custom）
    clip_pretrained_path: str = "openai"  # CLIP(ViT-B-32) 权重：open_clip 内置 tag（如 openai）或本地路径
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
    distill_align: str = 'roi_to_cls'     # 蒸馏对齐方式 ('roi_to_cls' / 'roi_to_pooled' / 'roi_to_pooled_attn')
    teacher_last_attn_type: str = 'qq'  # 教师最后一层注意力类型 ('qk' / 'qq' / 'qq+kk+vv')
    # FarSLIP 对齐：frozen=固定教师拷贝；ema=每步 EMA 同步教师；active=单模型（教师与学生同一模块）
    distill_type: str = 'frozen'  # 'frozen' | 'ema' | 'active'
    ema_momentum: float = 0.99   # 仅 distill_type=ema 时用于 teacher 动量更新
    distill_loss_weight: float = 0.5     # 蒸馏损失权重（loss_cos_sim 的系数）
    clip_loss_weight: float = 1.0        # CLIP 对比损失权重（loss_clip 的系数）
    use_clip_loss: bool = True           # 是否启用 clip_contrastive_loss（global_itc 路径）
    use_distill_loss: bool = True       # 是否启用 roi 蒸馏损失（loss_cos_sim）
    use_rafa: bool = False               # 是否启用 RaFA（global_itc 增强）
    use_hycd: bool = False               # 是否启用 HyCD（global_itc 增强）
    rafa_weight: float = 1.0             # RaFA 损失权重
    hycd_weight: float = 1.0             # HyCD 损失权重
    hycd_temperature: float = 1.0        # HyCD 温度系数 T
    hycd_alpha_blending: float = 0.5     # HyCD 混合标签系数 alpha
    rafa_prior_mu: float = 0.0           # RaFA 先验均值
    rafa_prior_sigma: float = 1.0        # RaFA 先验方差
    rafa_share_random_feat: bool = True  # RaFA 图文是否共享随机参考向量
    rafa_prior_stats_path: str = ""      # RaFA 先验统计量文件路径（.pt，包含 per-dim mu/sigma）
    
    # ============ 输出配置 ============
    output_dir: str = "./checkpoints"     # 模型 checkpoint 保存目录
    log_dir: str = "./logs"               # 日志文件保存目录
    tensorboard_dir: str = ""             # TensorBoard 事件目录；空字符串表示不启用
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


def _is_active_distill(config: TrainConfig) -> bool:
    return getattr(config, "distill_type", "frozen") == "active"


def build_models(config: TrainConfig, logger: logging.Logger):
    """
    构建 Teacher、Student 模型。

    - distill_type=active（FarSLIP active）：单模型，teacher_model 与 student_model 为同一对象。
    - frozen / ema：两次 create_model，teacher 全部冻结；ema 在训练循环中每步动量更新 teacher。
    """
    logger.info(f"Loading models from {config.pretrained_path}...")
    dt = getattr(config, "distill_type", "frozen")
    if dt not in ("frozen", "ema", "active"):
        raise ValueError(f"distill_type must be frozen|ema|active, got {dt}")

    if dt == "active":
        logger.info("\n=== distill-type=active: single CLIP (teacher == student, FarSLIP-style) ===")
        student_model = open_clip.create_model(
            config.model_name,
            pretrained=config.pretrained_path,
            precision=config.precision,
            device="cpu",
        )
        student_model.train()
        teacher_model = student_model
        nparam = sum(p.numel() for p in student_model.parameters())
        logger.info(f"✓ Single model: {type(student_model).__name__}, parameters={nparam:,} (trainable)")
        logger.info(
            "Note: SelfDistill allows frozen+clip_loss; FarSLIP forbids frozen+global_itc — not enforced here."
        )
        return teacher_model, student_model

    # ---- frozen / ema: separate frozen teacher copy ----
    logger.info("\n=== Loading Teacher Model (frozen copy) ===")
    teacher_model = open_clip.create_model(
        config.model_name,
        pretrained=config.pretrained_path,
        precision=config.precision,
        device="cpu",
    )
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    logger.info(f"✓ Teacher: {type(teacher_model).__name__}, frozen, params={sum(p.numel() for p in teacher_model.parameters()):,}")

    logger.info("\n=== Loading Student Model ===")
    student_model = open_clip.create_model(
        config.model_name,
        pretrained=config.pretrained_path,
        precision=config.precision,
        device="cpu",
    )
    student_model.train()
    logger.info(f"✓ Student: {type(student_model).__name__}, trainable, params={sum(p.numel() for p in student_model.parameters()):,}")

    if dt == "ema":
        logger.info(f"distill-type=ema: teacher will be updated each step (EMA momentum={config.ema_momentum})")
    else:
        logger.info("distill-type=frozen: teacher weights fixed (no EMA)")

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
    ds = getattr(config, "dataset", "rs3")
    if ds == "mgrs":
        mj = (getattr(config, "mgrs_json_path", "") or "").strip()
        mi = (getattr(config, "mgrs_image_root", "") or "").strip()
        if not mj or not mi:
            raise ValueError(
                "dataset=mgrs requires non-empty mgrs_json_path and mgrs_image_root "
                "(e.g. --mgrs-json /path/to/text_info.json --mgrs-image-root /path/to/global_imgs)"
            )
        return create_mgrs_dataloader(
            mgrs_json_path=mj,
            mgrs_image_root=mi,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            whole_image_size=config.whole_image_size,
            crop_size=config.crop_size,
            max_split=config.max_split,
            max_boxes=config.max_boxes,
            crop_scale=config.crop_scale,
            shuffle=shuffle and not config.distributed,
            drop_last=False,
            distributed=config.distributed,
            world_size=config.world_size,
            rank=config.rank,
            split=split,
            train_tar_count=config.train_tar_count,
            val_tar_count=config.val_tar_count,
        )

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
        split=split,
        train_tar_count=config.train_tar_count,
        val_tar_count=config.val_tar_count,
    )
    
    return dataloader


class DistillLoss(nn.Module):
    """
    知识蒸馏损失（包含两种独立的损失计算）
    
    1. cosine_distill_loss: 逐位置余弦相似度蒸馏损失（区域特征对齐）
    2. clip_contrastive_loss: CLIP 风格对比学习损失（全局图文匹配）
    """
    
    def __init__(self):
        super().__init__()

    def _contrastive_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: Optional[torch.Tensor] = None,
    ):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        sim = image_features @ text_features.t()
        if logit_scale is not None:
            sim = logit_scale * sim
        return sim, sim.t()
    
    def cosine_distill_loss(self, student_features: torch.Tensor, 
                           teacher_features: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """
        计算逐位置余弦相似度蒸馏损失
        
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
        
        # 归一化特征向量（L2 归一化）
        student_norm = F.normalize(student_valid, p=2, dim=-1)  # [N_valid, dim]
        teacher_norm = F.normalize(teacher_valid, p=2, dim=-1)  # [N_valid, dim]
        
        # 计算逐位置余弦相似度
        cosine_sim = (student_norm * teacher_norm).sum(dim=-1)  # [N_valid]
        
        # 损失 = 1 - 余弦相似度
        loss = (1 - cosine_sim).mean()
        
        return loss
    
    def clip_contrastive_loss(self, image_features: torch.Tensor, 
                              text_features: torch.Tensor,
                              logit_scale: torch.Tensor) -> torch.Tensor:
        """
        计算 CLIP 风格的对比损失
        
        Args:
            image_features: 图像特征 [B, D]
            text_features: 文本特征 [B, D]
            logit_scale: 温度系数（exp 后的结果，通常为 exp(4.6052) ≈ 100）
        
        Returns:
            loss: 对比损失标量值
        """
        logits_per_image, logits_per_text = self._contrastive_logits(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
        )
        
        # 3. 构建标签（对角线为正样本，即第 i 个样本匹配第 i 个样本）
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # 4. 计算对称交叉熵损失
        # 图像侧损失：每个图像对所有文本的相似度做交叉熵，正样本是对应文本
        loss_im = F.cross_entropy(logits_per_image, labels)
        # 文本侧损失：每个文本对所有图像的相似度做交叉熵，正样本是对应图像
        loss_txt = F.cross_entropy(logits_per_text, labels)
        
        # 5. 总损失：图像侧 + 文本侧 平均
        loss = (loss_im + loss_txt) / 2
        
        return loss

    def rafa_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        mu = 0.0,
        sigma = 1.0,
        share_random_feat: bool = True,
    ) -> torch.Tensor:
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        if isinstance(mu, torch.Tensor):
            mu = mu.to(device=image_features.device, dtype=image_features.dtype)
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device=image_features.device, dtype=image_features.dtype)
        if share_random_feat:
            rand_feat = torch.normal(
                mean=mu,
                std=sigma,
                size=image_features.shape,
                device=image_features.device,
                dtype=image_features.dtype,
            )
            rand_img = rand_feat
            rand_txt = rand_feat
        else:
            rand_img = torch.normal(
                mean=mu,
                std=sigma,
                size=image_features.shape,
                device=image_features.device,
                dtype=image_features.dtype,
            )
            rand_txt = torch.normal(
                mean=mu,
                std=sigma,
                size=text_features.shape,
                device=text_features.device,
                dtype=text_features.dtype,
            )
        return F.mse_loss(image_features, rand_img) + F.mse_loss(text_features, rand_txt)

    def hycd_loss(
        self,
        student_image_features: torch.Tensor,
        student_text_features: torch.Tensor,
        teacher_image_features: torch.Tensor,
        teacher_text_features: torch.Tensor,
        temperature: float = 1.0,
        alpha_blending: float = 0.5,
    ) -> torch.Tensor:
        temp = max(float(temperature), 1e-6)
        alpha = float(alpha_blending)
        alpha = min(max(alpha, 0.0), 1.0)
        s_i2t, s_t2i = self._contrastive_logits(student_image_features, student_text_features, logit_scale=None)
        with torch.no_grad():
            t_i2t, t_t2i = self._contrastive_logits(teacher_image_features, teacher_text_features, logit_scale=None)
            teacher_prob_i2t = F.softmax(t_i2t / temp, dim=-1)
            teacher_prob_t2i = F.softmax(t_t2i / temp, dim=-1)
            gt = torch.eye(t_i2t.shape[0], device=t_i2t.device, dtype=teacher_prob_i2t.dtype)
            hybrid_prob_i2t = alpha * gt + (1.0 - alpha) * teacher_prob_i2t
            hybrid_prob_t2i = alpha * gt + (1.0 - alpha) * teacher_prob_t2i
        kd_i2t = F.kl_div(
            F.log_softmax(s_i2t / temp, dim=-1),
            hybrid_prob_i2t,
            reduction="batchmean",
        ) * (temp ** 2)
        kd_t2i = F.kl_div(
            F.log_softmax(s_t2i / temp, dim=-1),
            hybrid_prob_t2i,
            reduction="batchmean",
        ) * (temp ** 2)
        return (kd_i2t + kd_t2i) / 2


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model

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
    logger: logging.Logger,
    tb_writer: Optional[Any] = None,
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
    active = _is_active_distill(config)
    if not active:
        teacher_model.eval()
    teacher_grad = active  # active: 教师即学生，需参与反传
    tctx = nullcontext() if teacher_grad else torch.no_grad()

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
        images, boxes_templates, image_crops_templates, masks, img_names, text_annotations = batch_data
        
        # 文本特征：frozen/ema 用 teacher 且 no_grad；active 用同一模型并保留梯度
        with tctx:
            if text_annotations is not None:
                text_strings = [t.decode('utf-8') if isinstance(t, bytes) else t for t in text_annotations]
                tokens = open_clip.tokenize(text_strings).to(config.device, non_blocking=True)
                text_features = _unwrap_model(teacher_model).encode_text(tokens)
            else:
                text_features = None
                tokens = None
        
        # 移动到设备
        images = images.to(config.device, non_blocking=True)
        boxes_templates = boxes_templates.to(config.device, non_blocking=True)
        image_crops_templates = image_crops_templates.to(config.device, non_blocking=True)
        masks = masks.to(config.device, non_blocking=True)
        
        # 前向传播（使用 AMP）
        with autocast(enabled=config.use_amp):
            student_base = _unwrap_model(student_model)
            if config.use_distill_loss:
                t_vis = _unwrap_model(teacher_model).visual
                with tctx:
                    if config.distill_align == "roi_to_pooled_attn":
                        teacher_features = t_vis.teacher_roi_encode(
                            image_crops_templates,
                            distill_align="roi_to_pooled",
                            last_attn_type=config.teacher_last_attn_type,
                        )
                        teacher_cls_features = t_vis.teacher_roi_encode(
                            image_crops_templates,
                            distill_align="roi_to_cls",
                            last_attn_type=config.teacher_last_attn_type,
                        )
                    else:
                        teacher_features = t_vis.teacher_roi_encode(
                            image_crops_templates,
                            distill_align=config.distill_align,
                            last_attn_type=config.teacher_last_attn_type,
                        )
                        teacher_cls_features = None

                # Student 编码（RoI 蒸馏需要 roi 特征）
                if hasattr(student_model, 'module'):
                    student_features, pooled = student_model.module.visual.student_encode(
                        images,
                        boxes_templates,
                        distill_align=config.distill_align,
                        teacher_cls_features=teacher_cls_features,
                    )
                else:
                    student_features, pooled = student_model.visual.student_encode(
                        images,
                        boxes_templates,
                        distill_align=config.distill_align,
                        teacher_cls_features=teacher_cls_features,
                    )
                loss_cos_sim = criterion.cosine_distill_loss(
                    student_features, teacher_features, masks.view(-1)
                )
                loss = config.distill_loss_weight * loss_cos_sim
            else:
                # 只做 global_itc：直接用 student 的全局 embedding
                pooled = student_base.encode_image(images, normalize=True)
                loss_cos_sim = torch.tensor(0.0, device=images.device)
                loss = config.distill_loss_weight * loss_cos_sim
            loss_clip = None
            if config.use_clip_loss:
                # 原始 CLIP 对比学习损失：pooled(图像全局) vs text_features(图本全局)
                loss_clip = criterion.clip_contrastive_loss(pooled, text_features, student_base.logit_scale)
                loss = loss + config.clip_loss_weight * loss_clip
            loss_rafa = None
            loss_hycd = None
            if config.use_rafa or config.use_hycd:
                student_text_features = student_base.encode_text(tokens, normalize=True)
                if config.use_rafa:
                    loss_rafa = criterion.rafa_loss(
                        image_features=pooled,
                        text_features=student_text_features,
                        mu=config.rafa_prior_mu,
                        sigma=config.rafa_prior_sigma,
                        share_random_feat=config.rafa_share_random_feat,
                    )
                    loss = loss + config.rafa_weight * loss_rafa
                if config.use_hycd:
                    with tctx:
                        teacher_global_features = _unwrap_model(teacher_model).encode_image(
                            images, normalize=True
                        )
                    loss_hycd = criterion.hycd_loss(
                        student_image_features=pooled,
                        student_text_features=student_text_features,
                        teacher_image_features=teacher_global_features,
                        teacher_text_features=text_features,
                        temperature=config.hycd_temperature,
                        alpha_blending=config.hycd_alpha_blending,
                    )
                    loss = loss + config.hycd_weight * loss_hycd
        
        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # 梯度裁剪（只对 student 模型）
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.grad_clip)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()

        # FarSLIP ema：每步用 student 动量更新 teacher（仅双模型）
        if (
            config.distill_type == "ema"
            and teacher_model is not student_model
        ):
            momentum = float(config.ema_momentum)
            s = _unwrap_model(student_model)
            t = _unwrap_model(teacher_model)
            with torch.no_grad():
                for p_q, p_k in zip(s.parameters(), t.parameters()):
                    p_k.data.mul_(momentum).add_(p_q.detach().data, alpha=1.0 - momentum)
                if hasattr(t, "logit_scale"):
                    t.logit_scale.clamp_(0, math.log(100))

        # 统计
        total_loss += loss.item()
        num_batches += 1
        
        # TensorBoard（仅 rank 0）
        if (
            tb_writer is not None
            and (not config.distributed or config.rank == 0)
            and (batch_idx + 1) % config.log_freq == 0
        ):
            step = global_step + batch_idx
            tb_writer.add_scalar("train/loss_batch", loss.item(), step)
            tb_writer.add_scalar("train/loss_distill", loss_cos_sim.detach().float().item(), step)
            if loss_clip is not None:
                tb_writer.add_scalar("train/loss_clip", loss_clip.detach().float().item(), step)
            if loss_rafa is not None:
                tb_writer.add_scalar("train/loss_rafa", loss_rafa.detach().float().item(), step)
            if loss_hycd is not None:
                tb_writer.add_scalar("train/loss_hycd", loss_hycd.detach().float().item(), step)
            tb_writer.add_scalar("train/loss_avg", total_loss / num_batches, step)
            tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            tb_writer.add_scalar("train/grad_norm", gn, step)
        
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
    
    if tb_writer is not None and (not config.distributed or config.rank == 0) and num_batches > 0:
        last_step = global_step + num_batches - 1
        tb_writer.add_scalar("train/epoch_avg_loss", avg_loss, last_step)
    
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
    logger: logging.Logger,
    tb_writer: Optional[Any] = None,
    tb_global_step: Optional[int] = None,
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
        images, boxes_templates, image_crops_templates, masks, img_names, text_annotations = batch_data
        
        if text_annotations is not None:
            text_strings = [t.decode('utf-8') if isinstance(t, bytes) else t for t in text_annotations]
            tokens = open_clip.tokenize(text_strings).to(config.device, non_blocking=True)
            text_features = _unwrap_model(teacher_model).encode_text(tokens)
        else:
            text_features = None

        images = images.to(config.device, non_blocking=True)
        boxes_templates = boxes_templates.to(config.device, non_blocking=True)
        image_crops_templates = image_crops_templates.to(config.device, non_blocking=True)
        masks = masks.to(config.device, non_blocking=True)

        student_base = _unwrap_model(student_model)
        t_vis = _unwrap_model(teacher_model).visual
        # forward
        if config.use_distill_loss:
            if config.distill_align == "roi_to_pooled_attn":
                teacher_features = t_vis.teacher_roi_encode(
                    image_crops_templates,
                    distill_align="roi_to_pooled",
                    last_attn_type=config.teacher_last_attn_type,
                )
                teacher_cls_features = t_vis.teacher_roi_encode(
                    image_crops_templates,
                    distill_align="roi_to_cls",
                    last_attn_type=config.teacher_last_attn_type,
                )
            else:
                teacher_features = t_vis.teacher_roi_encode(
                    image_crops_templates,
                    distill_align=config.distill_align,
                    last_attn_type=config.teacher_last_attn_type,
                )
                teacher_cls_features = None

            if hasattr(student_model, 'module'):
                student_features, pooled = student_model.module.visual.student_encode(
                    images,
                    boxes_templates,
                    distill_align=config.distill_align,
                    teacher_cls_features=teacher_cls_features,
                )
            else:
                student_features, pooled = student_model.visual.student_encode(
                    images,
                    boxes_templates,
                    distill_align=config.distill_align,
                    teacher_cls_features=teacher_cls_features,
                )

            loss_cos_sim = criterion.cosine_distill_loss(student_features, teacher_features, masks.view(-1))
            loss = config.distill_loss_weight * loss_cos_sim
        else:
            pooled = student_base.encode_image(images, normalize=True)
            loss_cos_sim = torch.tensor(0.0, device=images.device)
            loss = config.distill_loss_weight * loss_cos_sim
        loss_clip = None
        if config.use_clip_loss:
            loss_clip = criterion.clip_contrastive_loss(pooled, text_features, student_base.logit_scale)
            loss = loss + config.clip_loss_weight * loss_clip
        if config.use_rafa or config.use_hycd:
            student_text_features = student_base.encode_text(tokens, normalize=True)
            if config.use_rafa:
                loss_rafa = criterion.rafa_loss(
                    image_features=pooled,
                    text_features=student_text_features,
                    mu=config.rafa_prior_mu,
                    sigma=config.rafa_prior_sigma,
                    share_random_feat=config.rafa_share_random_feat,
                )
                loss = loss + config.rafa_weight * loss_rafa
            if config.use_hycd:
                teacher_global_features = _unwrap_model(teacher_model).encode_image(
                    images, normalize=True
                )
                loss_hycd = criterion.hycd_loss(
                    student_image_features=pooled,
                    student_text_features=student_text_features,
                    teacher_image_features=teacher_global_features,
                    teacher_text_features=text_features,
                    temperature=config.hycd_temperature,
                    alpha_blending=config.hycd_alpha_blending,
                )
                loss = loss + config.hycd_weight * loss_hycd

        total_loss += loss.item()
        num_batches += 1

        # 更新进度条信息
        if show_bar:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    val_time = time.time() - start_time

    logger.info(
        f"Validation | Epoch [{epoch+1}/{config.epochs}] | "
        f"Val Loss: {avg_loss:.4f} | "
        f"Batches: {num_batches} (len(dataloader)={len(dataloader)}) | "
        f"Time: {val_time:.2f}s"
    )

    if (
        tb_writer is not None
        and tb_global_step is not None
        and (not config.distributed or config.rank == 0)
    ):
        tb_writer.add_scalar("val/loss", avg_loss, tb_global_step)

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
    filename: str = 'checkpoint.pth',
    teacher_model: Optional[nn.Module] = None,
):
    """
    保存 checkpoint。student 始终保存；frozen/ema 双模型时额外保存 state_dict_teacher。
    """
    checkpoint_path = os.path.join(output_dir, filename)
    u_s = _unwrap_model(student_model)
    checkpoint = {
        'epoch': epoch,
        'state_dict': u_s.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    if teacher_model is not None:
        u_t = _unwrap_model(teacher_model)
        if u_t is not u_s:
            checkpoint['state_dict_teacher'] = u_t.state_dict()

    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")


def _strip_module_prefix_if_needed(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
    """若 ckpt 无 module. 前缀而模型为 DDP，或相反，尝试兼容加载。"""
    ckpt_keys = list(state_dict.keys())
    if not ckpt_keys:
        return state_dict
    has_module = ckpt_keys[0].startswith("module.")
    model_is_ddp = hasattr(model, "module")
    if has_module and not model_is_ddp:
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    if not has_module and model_is_ddp:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(
    student_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    logger: logging.Logger,
    teacher_model: Optional[nn.Module] = None,
) -> int:
    """加载 checkpoint；若存在 state_dict_teacher 且为双模型，则恢复 teacher。"""
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    sd = _strip_module_prefix_if_needed(checkpoint['state_dict'], student_model)
    student_model.load_state_dict(sd, strict=True)

    if (
        teacher_model is not None
        and 'state_dict_teacher' in checkpoint
        and _unwrap_model(teacher_model) is not _unwrap_model(student_model)
    ):
        tsd = _strip_module_prefix_if_needed(checkpoint['state_dict_teacher'], teacher_model)
        _unwrap_model(teacher_model).load_state_dict(tsd, strict=True)
        logger.info("✓ Loaded state_dict_teacher into teacher model")

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
    parser.add_argument('--dataset', type=str, default='rs3', choices=['rs3', 'mgrs'],
                        help='数据源：rs3=WebDataset .tar；mgrs=text_info.json + global_imgs（global_itc 仅用 brief_caption）')
    parser.add_argument('--mgrs-json', type=str, default='', dest='mgrs_json_path',
                        help='MGRS 的 text_info.json 路径（dataset=mgrs 时必填）')
    parser.add_argument('--mgrs-image-root', type=str, default='', dest='mgrs_image_root',
                        help='MGRS 图像根目录，对应 global_imgs（dataset=mgrs 时必填）')
    parser.add_argument('--rs3-tar-dir', type=str, default='./rs3',
                        help='RS3 train 文件目录')
    parser.add_argument('--rs3-val-dir', type=str, default='./rs3_val',
                        help='RS3 val 文件目录')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批次大小 (default: 4)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='数据加载进程数 (default: 2)')
    parser.add_argument('--whole-image-size', type=int, default=1024,
                        help='训练输入整图分辨率（正方形边长，default: 1024）')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='ROI/crop 输入分辨率（正方形边长，default: 224）')
    parser.add_argument('--max-split', type=int, default=7,
                        help='网格数据增强最大划分（与 TrainConfig.max_split 一致，default: 7）')
    parser.add_argument('--max-boxes', type=int, default=49,
                        help='单张图最大 RoI/box 数量（与 TrainConfig.max_boxes 一致，default: 49）')
    
    # 模型配置
    parser.add_argument('--model-name', type=str, default='ViT-B-32',
                        help='模型名称 (default: ViT-B-32)')
    parser.add_argument('--base-model-type', type=str, default='remoteclip',
                        choices=['remoteclip', 'georsclip', 'clip', 'custom'],
                        help='基础模型类型（default: remoteclip）。当不显式指定 --pretrained-path 时，用该类型选择权重路径。')
    parser.add_argument('--remoteclip-pretrained-path', type=str, default='/root/checkpoint/RemoteCLIP-ViT-B-32.pt',
                        help='RemoteCLIP(ViT-B-32) 权重路径（default: /root/checkpoint/RemoteCLIP-ViT-B-32.pt）')
    parser.add_argument('--georsclip-pretrained-path', type=str, default='/root/checkpoint/RS5M_ViT-B-32.pt',
                        help='GeoRSCLIP(ViT-B-32) 权重路径（default: /root/checkpoint/RS5M_ViT-B-32.pt）')
    parser.add_argument('--clip-pretrained-path', type=str, default='openai',
                        help='CLIP(ViT-B-32) 权重：open_clip 内置 tag（如 openai）或本地 .pt 路径（default: openai）')
    parser.add_argument('--pretrained-path', type=str, default='',
                        help='显式指定预训练权重路径（优先级最高）；留空则按 base-model-type 选择')
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
    parser.add_argument('--distill-align', type=str, default='roi_to_cls',
                        choices=['roi_to_cls', 'roi_to_pooled', 'roi_to_pooled_attn'],
                        help='蒸馏对齐方式: roi_to_cls / roi_to_pooled / roi_to_pooled_attn (default: roi_to_cls)')
    parser.add_argument('--teacher-last-attn-type', type=str, default='qq+kk+vv',
                        choices=['qk', 'qq', 'qq+kk+vv'],
                        help='教师最后一层注意力类型: qk / qq / qq+kk+vv (default: qq+kk+vv)')
    parser.add_argument('--distill-type', type=str, default='frozen',
                        choices=['frozen', 'ema', 'active'],
                        help='FarSLIP 对齐: frozen=固定教师拷贝(可与 clip_loss 并存); '
                             'ema=每步 EMA 从 student 更新 teacher; active=单模型 teacher==student')
    parser.add_argument('--ema-momentum', type=float, default=0.99,
                        help='distill-type=ema 时教师动量系数 (default: 0.99)')
    parser.add_argument('--distill-loss-weight', type=float, default=0.5,
                        help='distill loss 权重（loss_cos_sim 系数，default: 0.5）')
    parser.add_argument('--clip-loss-weight', type=float, default=1.0,
                        help='clip loss 权重（loss_clip 系数，default: 1.0）')
    parser.add_argument('--no-clip-loss', action='store_true',
                        help='禁用 clip_contrastive_loss（global_itc），RaFA/HyCD 仍可单独启用')
    parser.add_argument('--no-distill-loss', action='store_true',
                        help='禁用 roi 蒸馏损失（loss_cos_sim），仅做 global_itc / RaFA / HyCD')
    parser.add_argument('--use-rafa', action='store_true',
                        help='启用 RaFA（global_itc 额外正则）')
    parser.add_argument('--use-hycd', action='store_true',
                        help='启用 HyCD（global_itc 混合蒸馏）')
    parser.add_argument('--rafa-weight', type=float, default=1.0,
                        help='RaFA loss 权重（default: 1.0）')
    parser.add_argument('--hycd-weight', type=float, default=1.0,
                        help='HyCD loss 权重（default: 1.0）')
    parser.add_argument('--hycd-temperature', type=float, default=1.0,
                        help='HyCD 温度系数（default: 1.0）')
    parser.add_argument('--hycd-alpha-blending', type=float, default=0.5,
                        help='HyCD 混合标签系数 alpha（default: 0.5）')
    parser.add_argument('--rafa-prior-mu', type=float, default=0.0,
                        help='RaFA 随机先验均值（default: 0.0）')
    parser.add_argument('--rafa-prior-sigma', type=float, default=1.0,
                        help='RaFA 随机先验标准差（default: 1.0）')
    parser.add_argument('--rafa-prior-stats-path', type=str, default='',
                        help='RaFA 先验统计量文件路径（.pt，包含 mu/sigma: [D]）；提供后会覆盖 rafa_prior_mu/sigma')
    parser.add_argument('--rafa-share-random-feat', action='store_true',
                        help='RaFA 图文共享随机参考向量')
    parser.add_argument('--no-rafa-share-random-feat', dest='rafa_share_random_feat', action='store_false',
                        help='RaFA 图文不共享随机参考向量')
    parser.set_defaults(rafa_share_random_feat=True)
    parser.add_argument('--optimizer-betas', type=tuple, default=(0.9, 0.98),
                        help='AdamW 动量参数 (default: (0.9, 0.98))')
    parser.add_argument('--optimizer-eps', type=float, default=1e-8,
                        help='AdamW 数值稳定性参数 (default: 1e-8)')
    parser.add_argument('--val-tar-count', type=int, default=2,
                        help='验证集使用的 tar 文件数量 (default: 2, 约 6.25%%)')
    parser.add_argument('--train-tar-count', type=int, default=30,
                        help='训练集使用的 tar 文件数量（按排序取前N个，default: 30）')
    
    # 输出配置
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='模型保存目录 (default: ./checkpoints)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='日志文件保存目录 (default: ./logs)')
    parser.add_argument('--tensorboard-dir', type=str, default='',
                        help='TensorBoard 事件目录；留空则不启用 (例: ./logs/tb_run)')
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

    # Resolve pretrained weights path with priority:
    # 1) explicit --pretrained-path (non-empty)
    # 2) base-model-type mapping
    explicit_pretrained = (getattr(args, "pretrained_path", None) or "").strip()
    if explicit_pretrained:
        config.pretrained_path = explicit_pretrained
        config.base_model_type = "custom"
    else:
        bmt = str(getattr(args, "base_model_type", "remoteclip"))
        config.base_model_type = bmt
        if bmt == "remoteclip":
            config.pretrained_path = str(getattr(args, "remoteclip_pretrained_path")).strip()
        elif bmt == "georsclip":
            config.pretrained_path = str(getattr(args, "georsclip_pretrained_path")).strip()
        elif bmt == "clip":
            config.pretrained_path = str(getattr(args, "clip_pretrained_path", "openai")).strip()
        elif bmt == "custom":
            raise ValueError("base-model-type=custom requires --pretrained-path to be set.")
        else:
            raise ValueError(f"Unknown base-model-type: {bmt}")

    # argparse 使用 no-clip-loss 的形式：默认开启，传入后关闭
    if hasattr(args, "no_clip_loss") and args.no_clip_loss:
        config.use_clip_loss = False

    # argparse 使用 no-distill-loss 的形式：默认开启，传入后关闭
    if hasattr(args, "no_distill_loss") and args.no_distill_loss:
        config.use_distill_loss = False

    # Safety: if all global_itc-related losses are disabled, loss won't depend on student params.
    if (not config.use_distill_loss) and (not config.use_clip_loss) and (not config.use_rafa) and (not config.use_hycd):
        raise ValueError("All losses disabled: disable RaFA/HyCD/clip and distill simultaneously would make loss constant. Enable at least one of {distill_loss, clip_loss, rafa, hycd}.")
    
    # 特殊处理
    config.use_amp = not args.no_amp
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load RaFA prior stats if provided (per-dim mu/sigma), keep on CPU and move to device in loss call.
    if getattr(args, "rafa_prior_stats_path", ""):
        stats_path = str(args.rafa_prior_stats_path).strip()
        if stats_path:
            stats = torch.load(stats_path, map_location="cpu")
            if not ("mu" in stats and ("sigma" in stats or "std" in stats)):
                raise ValueError(f"Invalid RaFA stats file: {stats_path}. Expect keys: mu and sigma (or std).")
            mu_vec = stats["mu"]
            sigma_vec = stats.get("sigma", stats.get("std"))
            if not (isinstance(mu_vec, torch.Tensor) and isinstance(sigma_vec, torch.Tensor)):
                raise ValueError(f"Invalid RaFA stats file: {stats_path}. mu/sigma must be torch.Tensor.")
            config.rafa_prior_mu = mu_vec
            config.rafa_prior_sigma = sigma_vec
            config.rafa_prior_stats_path = stats_path
    
    # ==================== 创建输出目录 ====================
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # ==================== 设置日志 ====================
    logger = setup_logging(config.log_dir, rank=config.rank)
    logger.info("=" * 60)
    logger.info("RS3 Grid Distillation Training")
    logger.info("=" * 60)
    logger.info(
        f"Effective weights: base_model_type={config.base_model_type}, "
        f"pretrained_path={config.pretrained_path}"
    )
    logger.info(f"Data grid: max_split={config.max_split}, max_boxes={config.max_boxes}")
    logger.info(f"Configuration:")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # TensorBoard（仅 rank 0）
    tb_writer: Optional[Any] = None
    tb_dir = (getattr(config, "tensorboard_dir", None) or "").strip()
    if tb_dir:
        if SummaryWriter is None:
            logger.warning("未安装 tensorboard，无法写入 TensorBoard；请执行: pip install tensorboard")
        elif config.distributed and config.rank != 0:
            pass
        else:
            os.makedirs(tb_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"TensorBoard 已启用。查看: tensorboard --logdir {tb_dir}")
    
    # ==================== 构建模型 ====================
    logger.info("Building models...")
    teacher_model, student_model = build_models(config, logger)
    
    # 移动模型到设备
    teacher_model = teacher_model.to(config.device)
    student_model = student_model.to(config.device)
    logger.info(f"Models moved to device: {config.device}")
    logger.info(
        f"distill_type={config.distill_type} (ema_momentum={config.ema_momentum} when ema); "
        f"active => teacher is same module as student after DDP wrap"
    )

    # 分布式训练：只包装 student；active 时 teacher 引用同一 DDP 模块
    if config.distributed:
        if config.world_size <= 1 or config.rank >= config.world_size:
            logger.warning(f"Invalid distributed configuration: world_size={config.world_size}, rank={config.rank}")
            logger.warning("Falling back to single GPU training")
            config.distributed = False
        else:
            student_model = DDP(student_model, device_ids=[config.local_rank], find_unused_parameters=False)
            logger.info("Student model wrapped with DDP")
            if config.distill_type == "active":
                teacher_model = student_model
                logger.info("distill-type=active: teacher_model points to same DDP as student")
    
    # ==================== 构建数据加载器 ====================
    logger.info("\nBuilding dataloaders...")
    train_loader = build_dataloader(config, split='train', shuffle=True)
    val_loader = build_dataloader(config, split='val', shuffle=False)
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    
    # ==================== 构建优化器 ====================
    logger.info("\nBuilding optimizer...")
    
    # 优化器始终只包含 student（DDP 或裸模）；active 时与 teacher 共享参数
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.weight_decay
    )
    logger.info(f"Optimizer: AdamW (lr={config.lr}, betas={config.optimizer_betas}, "
                f"eps={config.optimizer_eps}, weight_decay={config.weight_decay})")
    logger.info("Optimizer: student parameters only (active mode shares weights with teacher)")
    
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
    criterion = DistillLoss()
    logger.info(f"Loss function: {args.loss_type}")
    
    # ==================== 构建 AMP Scaler ====================
    scaler = GradScaler(enabled=config.use_amp)
    logger.info(f"AMP enabled: {config.use_amp}")
    
    # ==================== 恢复训练（如果有 checkpoint） ====================
    start_epoch = 0
    if args.resume:
        if not config.distributed or config.rank == 0:
            sm = student_model.module if config.distributed else student_model
            tm = teacher_model.module if config.distributed and hasattr(teacher_model, "module") else teacher_model
            # active 时 teacher 与 student 同一模块，load_checkpoint 只写 student 即可
            start_epoch = load_checkpoint(sm, optimizer, args.resume, logger, teacher_model=tm)
    
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
            logger=logger,
            tb_writer=tb_writer,
        )
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Time: {train_metrics['time']:.2f}s")
        
        tb_step_end = (epoch + 1) * len(train_loader) - 1
        
        # 验证
        if val_loader is not None:
            val_metrics = validate(
                teacher_model=teacher_model,
                student_model=student_model,
                dataloader=val_loader,
                criterion=criterion,
                epoch=epoch,
                config=config,
                logger=logger,
                tb_writer=tb_writer,
                tb_global_step=tb_step_end,
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
                        filename='best_model.pth',
                        teacher_model=teacher_model,
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
                    filename=f'checkpoint_epoch_{epoch+1}.pth',
                    teacher_model=teacher_model,
                )
        
        if tb_writer is not None:
            tb_writer.flush()
    
    if tb_writer is not None:
        tb_writer.close()
    
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
