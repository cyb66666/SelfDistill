import os
import io
import json
import glob
import random
import logging
from PIL import Image
import webdataset as wds
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
import torch
from torch.nn.functional import pad
from src.open_clip.transform import image_transform, det_image_transform, AugmentationCfg
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
Image.MAX_IMAGE_PIXELS = None  # 解决大图片加载限制


class GridDistillImageMixin:
    """RS3 / MGRS 共用的网格 crop、box 模板与 read_image 逻辑。"""

    def _init_grid_common(self, transforms, max_split, crop_size, args):
        self.transforms = transforms
        self.args = args
        self.max_anns = args.max_boxes
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self._init_choices(max_split)
        self._init_boxes()

    def read_image(self, img_data):
        if img_data is None:
            return None
        width, height = img_data.size
        if width < 10 or height < 10:
            logging.warning(f"Invalid image, size {img_data.size}")
            return None
        return img_data

    def _init_choices(self, M=16):
        choices = []
        for m in range(1, M + 1):
            for n in range((m + 1) // 2, min(m * 2 + 1, M + 1)):
                choices.append((m, n))
        self.choices = choices

    def _init_boxes(self):
        box_templates = {}
        for choice in self.choices:
            M, N = choice
            grid_x, grid_y = torch.meshgrid(
                torch.linspace(0, 1, N + 1),
                torch.linspace(0, 1, M + 1),
                indexing='xy'
            )
            x0y0s = torch.stack([grid_x[:M, :N], grid_y[:M, :N]], dim=-1)
            x1y1s = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)
            pseudo_boxes = torch.cat([x0y0s, x1y1s], dim=-1).view(-1, 4)
            assert pseudo_boxes.shape[0] == M * N
            box_templates[choice] = pseudo_boxes
        self.box_templates = box_templates

    def _obtain_image_crops(self, image, choice):
        image_crops = []
        img_w, img_h = image.size
        normed_boxes = self.box_templates[choice]
        indices = list(range(len(normed_boxes)))
        random.shuffle(indices)
        indices = indices[:self.max_anns]
        boxes = normed_boxes * torch.tensor([img_w, img_h, img_w, img_h])
        for idx in indices:
            box = boxes[idx]
            x0, y0, x1, y1 = box.tolist()
            if self.args.crop_scale > 1.0:
                box_w, box_h = x1 - x0, y1 - y0
                cx, cy = (x1 + x0) / 2, (y1 + y0) / 2
                delta_factor = 0.5 * self.args.crop_scale
                x0, y0, x1, y1 = max(cx - box_w * delta_factor, 0), max(cy - box_h * delta_factor, 0), \
                    min(cx + box_w * delta_factor, img_w), min(cy + box_h * delta_factor, img_h)
            crop_transform = self.transforms[1] if len(self.transforms) > 1 else self.transforms[0]
            image_crops.append(crop_transform(image.crop((x0, y0, x1, y1))))
        return torch.stack(image_crops), normed_boxes[indices]


class RS3GridDistillDataset(Dataset, GridDistillImageMixin):
    """
    RS3 数据集的网格蒸馏数据加载器
    
    功能：
    1. 从 RS3 数据集的 tar 文件中加载遥感图像
    2. 将图像划分为多个网格区域（grid crops）
    3. 同时返回整图和裁剪区域的特征，用于知识蒸馏
    
    特点：
    - 使用 WebDataset 格式，支持高效流式加载
    - 迭代器方式访问，避免内存溢出
    - 支持变数量的网格区域，使用掩码标记有效区域
    """
    
    def __init__(self,
                 rs3_tar_dir,              # RS3 tar 文件所在目录
                 transforms,                 # 图像变换列表
                 max_split=4,               # 最大网格划分数量（16x16）
                 crop_size=224,              # 裁剪区域的尺寸
                 pre_transforms=False,       # 是否使用预变换（如数据增强）
                 args=None,                  # 其他配置参数
                 tar_pattern=None):          # tar 文件路径模式（用于数据集分割）
        """
        初始化数据集
        
        Args:
            rs3_tar_dir: RS3 tar 文件目录路径
            transforms: 图像变换列表，transforms[0] 用于整图变换
            max_split: 最大网格划分，决定最多可以生成多少个 crop 区域
            crop_size: 每个 crop 区域的输出尺寸
            pre_transforms: 是否使用预变换
            args: 包含 max_boxes, crop_scale 等参数的配置对象
        """
        self.rs3_tar_dir = rs3_tar_dir
        logging.debug(f'Loading RS3 data from {rs3_tar_dir}.')
        self._init_grid_common(transforms, max_split, crop_size, args)

        # 设置 RS3 tar 文件路径模式
        if tar_pattern is None:
            tar_pattern = sorted(glob.glob(os.path.join(rs3_tar_dir, "*.tar")))
        self.tar_files = tar_pattern
        
        # 构建 WebDataset 数据集（使用传入的 tar_pattern）
        # 注意：保持为迭代器形式，不要转为列表，避免内存溢出
        # WebDataset 默认 workersplitter=split_by_worker：按 DataLoader worker id 对 shard 列表 stride 分配；
        # 若 DataLoader num_workers > len(tar)，部分 worker 分不到 shard，会提前结束（验证常只跑极少 batch）。
        # 请在 create_rs3_dataloader 里保证 num_workers <= len(tar_pattern)。
        self.dataset = (
            wds.WebDataset(tar_pattern, nodesplitter=None, empty_check=False)
            .decode(self.rs3_wds_decoder)  # 使用自定义解码器
            .rename(
                img_content="img_content",  # 图片内容
                img_name="img_name",        # 图片名称
                txt="caption"               # 文本注释（RS3 中使用 caption 字段）
            )
        )
        
        logging.debug(f'Initialized RS3 dataset with pattern: {tar_pattern}')
        
        # 预变换（如数据增强）
        if pre_transforms:
            self.pre_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip()])
        else:
            self.pre_transforms = None
    
    def rs3_wds_decoder(self, key, value):
        """
        自定义 WebDataset 解码器
        
        功能：将字节流解码为 Python 对象
        
        Args:
            key: 数据键名（如 .img_content, .img_name, .txt）
            value: 数据值（字节流）
        
        Returns:
            解码后的对象（PIL Image、字符串或 None）
        """
        if key.endswith(".img_content"):
            # 图片内容：字节流 -> PIL Image
            assert isinstance(value, bytes)
            img = Image.open(io.BytesIO(value)).convert("RGB")
            return img
        elif key.endswith(".img_name"):
            # 图片名称：字节流 -> 字符串
            return value.decode("utf-8")
        elif key.endswith(".txt"):
            # 文本注释：字节流 -> 字符串
            return value.decode("utf-8")
        return value  # 其他字段直接返回
    
    def __len__(self):
        """
        返回数据集大小（预估值）
        
        注意：WebDataset 是流式的，无法精确获取总样本数
        这里返回一个预估值，实际使用时可能不需要精确长度
        
        Returns:
            预估的样本数量
        """
        # 估算数据量（WebDataset 无法精确获取长度）
        # 可以根据 tar 文件数量 * 每个 tar 的平均图片数估算
        n = max(len(getattr(self, "tar_files", [])), 1)
        return 3400 * n  # 预估值
    
    def __getitem__(self, idx):
        """
        获取一个样本数据
        
        使用迭代器方式从 WebDataset 中获取下一个样本
        支持自动重试（如果图片加载失败）
        
        Args:
            idx: 样本索引（未实际使用，WebDataset 是流式的）
        
        Returns:
            tuple: 包含以下元素的元组：
                - new_image: 整图张量 [3, 1024, 1024]
                    * RGB 三通道遥感图像
                    * 已 resize 到 1024×1024
                    * 已归一化（mean=[0.3759, 0.3912, 0.3618], std=[0.2582, 0.2472, 0.2461]）
                    * 值范围 [0, 1] 的 tensor
                
                - boxes_template: 边界框模板 [max_anns, 5]
                    * 前 4 列：归一化坐标 [x0, y0, x1, y1]，范围 [0, 1]
                        - (x0, y0): 左上角坐标
                        - (x1, y1): 右下角坐标
                    * 第 5 列：有效标志（1.0=有效区域，0.0=填充区域）
                    * 前 num_valid_boxes 行为真实裁剪区域，其余为填充
                
                - image_crops_template: 裁剪区域张量 [max_anns, 3, 224, 224]
                    * 从原图裁剪出的网格区域
                    * 每个 crop 已 resize 到 224×224
                    * 已归一化（与整图相同的均值和标准差）
                    * 前 num_valid_boxes 个为真实裁剪，其余为零填充
                
                - mask_template: 有效区域掩码 [max_anns]
                    * bool 类型 tensor
                    * True: 对应位置是有效的裁剪区域
                    * False: 对应位置是填充区域
                    * 用于损失计算时屏蔽无效的 padding 部分
                
                - image_name: 图片名称（字符串）
                    * RS3 数据集中的唯一标识符
                    * 格式如 "RS3-1024-000001"
                
                - text_annotation: 文本注释（字符串或 None）
                    * 图像的文本描述/标注
                    * 如果数据集中包含 .txt 文件则加载
                    * 如果没有文本注释则为 None
        """
        # WebDataset 作为迭代器，需要通过迭代来获取数据
        # 对于训练用途，可以配合 DataLoader 的 worker_init_fn 使用
        
        # 创建迭代器（每个 worker 会有自己的迭代器）
        if not hasattr(self, '_dataset_iter') or self._dataset_iter is None:
            self._dataset_iter = iter(self.dataset)
        
        try:
            sample = next(self._dataset_iter)
        except StopIteration:
            # 如果迭代器耗尽，重新创建
            self._dataset_iter = iter(self.dataset)
            sample = next(self._dataset_iter)
        
        old_image = sample['img_content']
        image_name = sample['img_name']
        text_annotation = sample.get('txt', None)  # 获取文本注释（可选）
        
        # 验证并加载图片
        old_image = self.read_image(old_image)
        if old_image is None:
            # 如果图片加载失败，获取下一个
            return self.__getitem__(idx + 1)
        
        # 应用整图变换
        new_image = self.transforms[0](old_image)  # transforms[0]: 整图变换 -> [3, 1024, 1024]
        
        # 计算缩放比例
        scale = get_scale(old_image, new_image)
        
        # 创建模板（填充到固定大小）
        boxes_template = torch.zeros(self.max_anns, 4 + 1)    # xyxy + 有效标志
        image_crops_template = torch.zeros(self.max_anns, 3, *self.crop_size)
        mask_template = torch.zeros(self.max_anns, dtype=torch.bool)  # 掩码
        
        # 获取裁剪区域（boxes 已经是归一化比例，不需要再缩放）
        image_crops, boxes = self._obtain_image_crops(old_image,
                                                      random.choice(self.choices))
        assert image_crops.shape[0] == boxes.shape[0]
        _, h, w = new_image.shape
        
        # 实际有效的 box 数量
        num_valid_boxes = boxes.shape[0]
        
        # 填充模板
        boxes_template[:num_valid_boxes, :4] = boxes
        boxes_template[:num_valid_boxes, 4] = 1.0
        
        image_crops_template[:num_valid_boxes] = image_crops
        
        # 设置掩码：前 num_valid_boxes 个位置为 True，其余为 False
        mask_template[:num_valid_boxes] = True
        
        return new_image, boxes_template, image_crops_template, mask_template, image_name, text_annotation


class MGRSGridDistillDataset(Dataset, GridDistillImageMixin):
    """
    MGRS（text_info.json + global_imgs）网格蒸馏数据集。
    全局文本仅使用 brief_caption，与 RS3GridDistillDataset 返回相同六元组。
    """

    def __init__(
        self,
        records,
        image_root,
        transforms,
        max_split=4,
        crop_size=224,
        pre_transforms=False,
        args=None,
        max_skip_attempts: int = 256,
    ):
        self.records = records
        self.image_root = image_root
        self.max_skip_attempts = max(1, int(max_skip_attempts))
        self._init_grid_common(transforms, max_split, crop_size, args)
        if pre_transforms:
            self.pre_transforms = T.Compose([T.RandomHorizontalFlip()])
        else:
            self.pre_transforms = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        n = len(self.records)
        max_tries = min(self.max_skip_attempts, n)
        for attempt in range(max_tries):
            i = (idx + attempt) % n
            item = self.records[i]
            rel = item.get("global_filepath")
            if not rel:
                continue
            brief = item.get("brief_caption")
            if brief is None:
                continue
            if isinstance(brief, bytes):
                brief = brief.decode("utf-8", errors="replace")
            text_annotation = str(brief).strip()
            if not text_annotation:
                continue
            full_path = os.path.join(self.image_root, rel)
            try:
                old_image = Image.open(full_path).convert("RGB")
            except Exception as e:
                logging.debug("MGRS skip unreadable image %s: %s", full_path, e)
                continue
            old_image = self.read_image(old_image)
            if old_image is None:
                continue
            if self.pre_transforms is not None:
                old_image = self.pre_transforms(old_image)
            new_image = self.transforms[0](old_image)
            boxes_template = torch.zeros(self.max_anns, 4 + 1)
            image_crops_template = torch.zeros(self.max_anns, 3, *self.crop_size)
            mask_template = torch.zeros(self.max_anns, dtype=torch.bool)
            image_crops, boxes = self._obtain_image_crops(old_image, random.choice(self.choices))
            assert image_crops.shape[0] == boxes.shape[0]
            num_valid_boxes = boxes.shape[0]
            boxes_template[:num_valid_boxes, :4] = boxes
            boxes_template[:num_valid_boxes, 4] = 1.0
            image_crops_template[:num_valid_boxes] = image_crops
            mask_template[:num_valid_boxes] = True
            image_name = rel
            return new_image, boxes_template, image_crops_template, mask_template, image_name, text_annotation
        raise RuntimeError(
            f"MGRSGridDistillDataset: 从索引 {idx} 起连续尝试 {max_tries} 条仍无有效样本（需磁盘上可读图像 + brief_caption）。"
            f"请检查 mgrs_image_root={self.image_root!r} 是否已解压 global_imgs，并与 text_info.json 中 global_filepath 一致。"
        )


def get_scale(old_image, new_image):
    """计算图像缩放比例"""
    old_w, old_h = old_image.size
    new_h, new_w = new_image.shape[1:]
    # 返回 4 个维度的缩放比例 (x0, y0, x1, y1)
    scale_x = old_w / new_w
    scale_y = old_h / new_h
    return torch.tensor([scale_x, scale_y, scale_x, scale_y])


def verify_mgrs_image_root(mgrs_image_root: str, records: list, probe: int = 512) -> None:
    """
    主进程启动时校验：图像目录存在，且能在若干条记录内成功打开至少一张图。
    避免 DataLoader worker 内对整表线性扫描。
    """
    root = os.path.abspath(mgrs_image_root)
    if not os.path.exists(root):
        hint = ""
        parent = os.path.dirname(root)
        if os.path.basename(root) == "global_imgs" and os.path.isdir(parent):
            part_globs = sorted(glob.glob(os.path.join(parent, "global_imgs.tar.part-*")))
            if part_globs:
                hint = (
                    f"\n在 {parent!r} 下检测到 global_imgs.tar.part-* 分卷但尚未解压出 global_imgs 目录，可执行：\n"
                    f"  cd {parent!r} && cat global_imgs.tar.part-* > global_imgs.tar && tar -xf global_imgs.tar\n"
                    f"然后将 --mgrs-image-root 指向解压得到的 global_imgs。"
                )
        raise FileNotFoundError(
            f"MGRS 图像根目录不存在: {mgrs_image_root!r}（解析为 {root!r}）。"
            f"请将其设为已解压的 global_imgs 文件夹。{hint}"
        )
    if not os.path.isdir(root):
        raise NotADirectoryError(
            f"MGRS mgrs_image_root 不是目录: {mgrs_image_root!r}"
        )
    parent = os.path.dirname(root)
    if os.path.isdir(parent):
        part_globs = sorted(glob.glob(os.path.join(parent, "global_imgs.tar.part-*")))
        if part_globs and len(os.listdir(root)) == 0:
            raise RuntimeError(
                f"目录 {root!r} 存在但为空；同目录下存在分卷 {os.path.basename(part_globs[0])} 等。"
                f"请先合并并解压，例如：\n"
                f"  cd {parent!r} && cat global_imgs.tar.part-* > global_imgs.tar && tar -xf global_imgs.tar\n"
                f"然后将 --mgrs-image-root 指向解压出的 global_imgs。"
            )
    n_probe = min(probe, len(records))
    for k in range(n_probe):
        item = records[k]
        rel = item.get("global_filepath")
        if not rel:
            continue
        brief = item.get("brief_caption")
        if brief is None:
            continue
        if isinstance(brief, bytes):
            brief = brief.decode("utf-8", errors="replace")
        if not str(brief).strip():
            continue
        full_path = os.path.join(root, rel)
        if not os.path.isfile(full_path):
            continue
        try:
            with Image.open(full_path) as im:
                im.convert("RGB")
            logging.info(
                "MGRS 图像校验通过：已打开 %s（探测第 %d/%d 条）",
                full_path,
                k + 1,
                n_probe,
            )
            return
        except Exception as e:
            logging.debug("MGRS probe skip %s: %s", full_path, e)
            continue
    example_rel = next((r.get("global_filepath") for r in records[:20] if r.get("global_filepath")), "<unknown>")
    example_path = os.path.join(root, example_rel) if example_rel != "<unknown>" else root
    part_hint = ""
    if os.path.isdir(parent):
        part_globs = sorted(glob.glob(os.path.join(parent, "global_imgs.tar.part-*")))
        if part_globs:
            part_hint = (
                f" 同目录下存在 global_imgs 分卷包，若尚未解压，请："
                f"cd {parent!r} && cat global_imgs.tar.part-* > global_imgs.tar && tar -xf global_imgs.tar\n"
            )
    raise RuntimeError(
        f"MGRS：在 mgrs_image_root={root!r} 下对前 {n_probe} 条记录探测后，仍无法打开任何有效图像（需同时有 brief_caption）。"
        f"示例期望路径: {example_path!r}。\n{part_hint}"
        f"请确认已解压 global_imgs、且 --mgrs-image-root 指向该目录（JSON 中 global_filepath 相对此目录）。"
    )


def split_mgrs_records(data_list, train_tar_count, val_tar_count):
    """
    将 MGRS JSON 列表按 global_filepath 排序后，均分为 S=train+val 个连续块；
    训练集为前 train_tar_count 块拼接，验证集为剩余 val_tar_count 块拼接。
    """
    if train_tar_count < 0 or val_tar_count < 0:
        raise ValueError("train_tar_count and val_tar_count must be non-negative")
    S = train_tar_count + val_tar_count
    if S <= 0:
        raise ValueError("train_tar_count + val_tar_count must be positive")
    sorted_list = sorted(data_list, key=lambda x: str(x.get("global_filepath", "")))
    n = len(sorted_list)
    boundaries = [(i * n) // S for i in range(S + 1)]
    chunks = [sorted_list[boundaries[i]: boundaries[i + 1]] for i in range(S)]
    train_records = []
    for j in range(train_tar_count):
        train_records.extend(chunks[j])
    val_records = []
    for j in range(train_tar_count, S):
        val_records.extend(chunks[j])
    return train_records, val_records


def create_mgrs_dataloader(
    mgrs_json_path="./data/mgrs/text_info.json",
    mgrs_image_root="./data/mgrs/global_imgs",
    batch_size=4,
    num_workers=2,
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
    split="train",
    train_tar_count=9,
    val_tar_count=1,
):
    """
    从 MGRS text_info.json 创建 DataLoader（Map-style + 可选 DistributedSampler）。
    """
    class Args:
        def __init__(self, max_boxes, crop_scale, train_ratio):
            self.max_boxes = max_boxes
            self.crop_scale = crop_scale
            self.train_ratio = train_ratio

    args = Args(
        max_boxes=max_boxes,
        crop_scale=crop_scale,
        train_ratio=1.0,
    )

    whole_image_transform = T.Compose([
        T.Resize((whole_image_size, whole_image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.3759, 0.3912, 0.3618],
            std=[0.2582, 0.2472, 0.2461],
        ),
    ])
    crop_transform = T.Compose([
        T.Resize((crop_size, crop_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.3759, 0.3912, 0.3618],
            std=[0.2582, 0.2472, 0.2461],
        ),
    ])

    if not os.path.isfile(mgrs_json_path):
        raise FileNotFoundError(f"MGRS JSON not found: {mgrs_json_path}")
    with open(mgrs_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"MGRS JSON root must be a list, got {type(raw)}")

    train_records, val_records = split_mgrs_records(raw, train_tar_count, val_tar_count)
    if split == "train":
        records = train_records
    elif split == "val":
        records = val_records
    else:
        raise ValueError(f"Unknown split: {split}")

    if len(records) == 0:
        raise RuntimeError(
            f"MGRS split={split} is empty (total={len(raw)}, "
            f"train_tar_count={train_tar_count}, val_tar_count={val_tar_count})"
        )

    verify_mgrs_image_root(mgrs_image_root, records)

    dataset = MGRSGridDistillDataset(
        records=records,
        image_root=mgrs_image_root,
        transforms=[whole_image_transform, crop_transform],
        max_split=max_split,
        crop_size=crop_size,
        pre_transforms=False,
        args=args,
    )

    sampler = None
    if distributed:
        if world_size < 1:
            logging.warning(f"Invalid world_size ({world_size}), disabling distributed sampling")
            distributed = False
        elif rank >= world_size:
            logging.warning(f"Invalid rank ({rank}) for world_size ({world_size}), disabling distributed sampling")
            distributed = False
        else:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
            )
            shuffle = False
            drop_last = True

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
        persistent_workers=num_workers > 0,
    )
    return dataloader


def create_rs3_dataloader(
    rs3_tar_dir="./rs3",
    batch_size=4,
    num_workers=2,
    whole_image_size=1024,
    crop_size=224,
    max_split=4,
    max_boxes=16,
    crop_scale=1.0,
    shuffle=True,
    drop_last=False,          # 'train' 或 'val'
    distributed=False,          # 是否使用分布式训练
    world_size=1,               # 分布式训练的总进程数
    rank=0,                      # 当前进程的 rank
    split='train',
    train_tar_count=30,
    val_tar_count=2,
):
    """
    创建 RS3 数据集的 DataLoader
    
    Args:
        rs3_tar_dir: RS3 tar 文件目录
        batch_size: 批次大小
        num_workers: 数据加载的并行工作进程数
        whole_image_size: 整图 resize 尺寸（默认 1024x1024）
        crop_size: crop 区域输出尺寸（默认 224x224）
        max_split: 最大网格划分（默认 4，对应 4×4=16 个网格）
        max_boxes: 最大网格区域数量（用于填充，默认 16）
        crop_scale: crop 区域的缩放比例（>1.0 表示扩大，默认 1.0）
        shuffle: 是否打乱数据（训练时设为 True）
        drop_last: 是否丢弃最后不完整的批次
        split: 数据集分割 ('train' 或 'val')
            - 'train': 使用前 (32-val_tar_count) 个 tar 文件
            - 'val': 使用最后 val_tar_count 个 tar 文件
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    # 模拟 args 配置
    class Args:
        def __init__(self, max_boxes, crop_scale, train_ratio):
            self.max_boxes = max_boxes
            self.crop_scale = crop_scale
            self.train_ratio = train_ratio
    
    args = Args(
        max_boxes=max_boxes,
        crop_scale=crop_scale,
        train_ratio=1.0
    )
    
    # 定义图像变换
    # 1. 整图变换：resize 到 whole_image_size，然后归一化
    whole_image_transform = T.Compose([
        T.Resize((whole_image_size, whole_image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.3759, 0.3912, 0.3618],  # RS3 数据集的通道均值
            std=[0.2582, 0.2472, 0.2461]    # RS3 数据集的通道标准差
        )
    ])
    
    # 2. Crop 变换：从原图裁剪后 resize 到 crop_size
    crop_transform = T.Compose([
        T.Resize((crop_size, crop_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.3759, 0.3912, 0.3618],
            std=[0.2582, 0.2472, 0.2461]
        )
    ])
    
    all_tar_files = sorted(glob.glob(os.path.join(rs3_tar_dir, "*.tar")))
    if len(all_tar_files) == 0:
        raise RuntimeError(f"No .tar files found under: {rs3_tar_dir}")

    if split == "train":
        tar_pattern = all_tar_files[:train_tar_count]
    elif split == "val":
        tar_pattern = all_tar_files[-val_tar_count:]
    else:
        raise ValueError(f"Unknown split: {split}")

    if len(tar_pattern) == 0:
        raise RuntimeError(
            f"No tar files selected for split={split}. total={len(all_tar_files)}, "
            f"train_tar_count={train_tar_count}, val_tar_count={val_tar_count}"
        )

    # 创建数据集
    dataset = RS3GridDistillDataset(
        rs3_tar_dir=rs3_tar_dir,
        transforms=[whole_image_transform, crop_transform],
        max_split=max_split,
        crop_size=crop_size,
        pre_transforms=False,
        args=args,
        tar_pattern=tar_pattern
   )
   
   # 创建 DataLoader
   # 分布式训练时使用 DistributedSampler
    sampler= None
    
    # WebDataset 默认按 worker 均分 shard（tar）：worker id 对 num_workers 取 stride，无 shard 的 worker 会空转失败，
    # DataLoader 往往在跑过前几个 batch 后就停止（例如 val 只有 2 个 tar、num_workers=8 时约 2 个 batch 就停）。
    n_shards = len(tar_pattern)
    actual_num_workers = num_workers
    if actual_num_workers > 0 and n_shards > 0 and actual_num_workers > n_shards:
        logging.warning(
            f"split={split}: WebDataset 仅有 {n_shards} 个 shard(.tar)，num_workers={num_workers} 会导致 "
            f"{actual_num_workers - n_shards} 个 worker 分不到数据、验证/训练提前结束。"
            f"已将 num_workers 限制为 {n_shards}。"
        )
        actual_num_workers = n_shards
    
    if distributed:
        # 验证 rank 和 world_size 是否有效
        if world_size < 1:
            logging.warning(f"Invalid world_size ({world_size}), disabling distributed sampling")
            distributed = False
        elif rank >= world_size:
            logging.warning(f"Invalid rank ({rank}) for world_size ({world_size}), disabling distributed sampling")
            distributed = False
        else:
            sampler= DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
            shuffle = False  # 使用 sampler 时不能设置 shuffle
            drop_last = True  # 分布式训练时通常丢弃最后一个不完整的 batch
   
    dataloader= DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=actual_num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
        persistent_workers=actual_num_workers > 0 if actual_num_workers > 0 else False  # 启用持久化 workers 加速
    )
   
    return dataloader

# -------------------------- 测试加载 --------------------------
if __name__ == "__main__":
    
    
    # ==================== 配置参数 ====================
    RS3_TAR_DIR = "./rs3"      # RS3 数据集 tar 文件所在目录
    BATCH_SIZE = 4             # 每个批次的样本数量
    NUM_WORKERS = 2            # 数据加载的并行工作进程数
    
    print("正在测试 RS3GridDistillDataset...")
    
    # ==================== 创建数据集 ====================
    dataloader = create_rs3_dataloader(
        rs3_tar_dir=RS3_TAR_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        whole_image_size=1024,  # 整图 1024×1024
        crop_size=224,          # crop 224×224
        max_split=4,            # 最大 4×4=16 个网格
        max_boxes=16,           # 匹配 max_split
        crop_scale=1.0,
        shuffle=True,
        drop_last=False
    )
    
    # ==================== 遍历数据集 ====================
    # DataLoader 会自动调用 dataset 的 __getitem__ 方法获取数据
    for batch_idx, batch_data in enumerate(dataloader):
        # 解包返回的数据
        # images: 整图张量 [batch_size, 3, 1024, 1024]
        # boxes_templates: 边界框模板 [batch_size, max_boxes, 5] (xyxy + 有效标志)
        # image_crops_templates: 裁剪区域张量 [batch_size, max_boxes, 3, 224, 224]
        # masks: 有效区域掩码 [batch_size, max_boxes] (True 表示有效，False 表示填充)
        # img_names: 图片名称列表
        # text_annotations: 文本注释列表（如果有）
        if len(batch_data) == 6:
            # 包含文本注释的版本
            images, boxes_templates, image_crops_templates, masks, img_names, text_annotations = batch_data
        else:
            # 不包含文本注释的旧版本
            images, boxes_templates, image_crops_templates, masks, img_names = batch_data
            text_annotations = None
        
        print(f"\n批次 {batch_idx + 1}:")
        print(f"  图片张量形状：{images.shape}")
        print(f"  boxes_template 形状：{boxes_templates.shape}")
        print(f"  image_crops_template 形状：{image_crops_templates.shape}")
        print(f"  mask 形状：{masks.shape}")
        print(f"  图片数量：{len(img_names)}")
        print(f"  第一张图片名：{img_names[0]}")
        print(f"  第一张图片的有效区域数：{masks[0].sum().item()}")
        if text_annotations is not None:
            print(f"  第一个文本注释：{text_annotations[0][:100]}...")  # 只显示前 100 字符
        
        # 只显示前 2 个批次
        if batch_idx >= 1:
            break
    
    print("\n测试完成！")