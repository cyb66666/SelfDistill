"""
Compute per-dimension Gaussian prior stats (mu/std) for RaFA from RS3 training data.

We compute statistics on L2-normalized teacher features to match current RaFA usage
in train_distill.py (which normalizes features before applying RaFA loss).

Modes:
  - image: teacher.encode_image(images, normalize=True)
  - text:  teacher.encode_text(tokens, normalize=True)
  - both:  update stats with both image and text features for each sample pair
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from tqdm import tqdm

from src import open_clip
from src.training.data import create_rs3_dataloader


@dataclass
class RunningStats:
    """Welford running mean/variance for vectors."""
    n: int
    mean: torch.Tensor
    m2: torch.Tensor

    @classmethod
    def create(cls, dim: int, device: torch.device, dtype: torch.dtype):
        mean = torch.zeros(dim, device=device, dtype=dtype)
        m2 = torch.zeros(dim, device=device, dtype=dtype)
        return cls(n=0, mean=mean, m2=m2)

    def update(self, x: torch.Tensor):
        """
        x: [B, D]
        Updates in float64 for numerical stability, stores back in original dtype.
        """
        if x.numel() == 0:
            return
        x64 = x.detach().to(torch.float64)
        mean64 = self.mean.to(torch.float64)
        m264 = self.m2.to(torch.float64)
        for i in range(x64.shape[0]):
            self.n += 1
            delta = x64[i] - mean64
            mean64 = mean64 + delta / self.n
            delta2 = x64[i] - mean64
            m264 = m264 + delta * delta2
        self.mean = mean64.to(self.mean.dtype)
        self.m2 = m264.to(self.m2.dtype)

    def finalize(self, eps: float = 1e-6):
        if self.n < 2:
            var = torch.ones_like(self.mean)
        else:
            var = self.m2 / (self.n - 1)
        std = torch.sqrt(torch.clamp(var, min=eps))
        return self.mean, std


def _build_dataloader(args):
    # mimic TrainConfig defaults used in train_distill.py
    dataloader = create_rs3_dataloader(
        rs3_tar_dir=args.rs3_tar_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        whole_image_size=args.whole_image_size,
        crop_size=args.crop_size,
        max_split=args.max_split,
        max_boxes=args.max_boxes,
        crop_scale=args.crop_scale,
        shuffle=False,
        drop_last=False,
        distributed=False,
        world_size=1,
        rank=0,
        split="train",
        train_tar_count=args.train_tar_count,
        val_tar_count=0,
    )
    return dataloader


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Compute RaFA prior stats (per-dim mu/std) from RS3 teacher features.")
    parser.add_argument("--rs3-tar-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained-path", type=str, required=True)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--mode", type=str, default="both", choices=["image", "text", "both"])
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max number of (image,text) pairs to process. For mode=both, stats are updated with both image and text features per pair.")
    parser.add_argument("--save-path", type=str, required=True)

    # dataloader knobs (keep consistent with train script defaults)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--whole-image-size", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-split", type=int, default=4)
    parser.add_argument("--max-boxes", type=int, default=16)
    parser.add_argument("--crop-scale", type=float, default=1.0)
    parser.add_argument("--train-tar-count", type=int, default=30)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("compute_rafa_prior_stats")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info(f"Loading teacher: model={args.model_name}, pretrained={args.pretrained_path}, device={device}")
    teacher = open_clip.create_model(
        args.model_name,
        pretrained=args.pretrained_path,
        precision=args.precision,
        device="cpu",
    )
    teacher.eval()
    teacher = teacher.to(device)

    dl = _build_dataloader(args)

    stats: Optional[RunningStats] = None
    n_pairs = 0
    pbar = tqdm(dl, desc=f"Stats({args.mode})", total=len(dl))
    for batch in pbar:
        if n_pairs >= args.max_samples:
            break
        # batch: images, boxes_templates, image_crops_templates, masks, img_names, text_annotations
        if len(batch) == 6:
            images, _, _, _, _, text_annotations = batch
        else:
            images, _, _, _, _ = batch
            text_annotations = None

        images = images.to(device, non_blocking=True)

        # tokenize text (may be bytes)
        if text_annotations is not None:
            text_strings = [t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else str(t) for t in text_annotations]
            tokens = open_clip.tokenize(text_strings).to(device, non_blocking=True)
        else:
            tokens = None

        # compute features (normalized)
        img_feat = None
        txt_feat = None
        if args.mode in ("image", "both"):
            img_feat = teacher.encode_image(images, normalize=True)  # [B, D]
        if args.mode in ("text", "both"):
            if tokens is None:
                continue
            txt_feat = teacher.encode_text(tokens, normalize=True)  # [B, D]

        # init stats
        if stats is None:
            base = img_feat if img_feat is not None else txt_feat
            if base is None:
                continue
            dim = base.shape[-1]
            stats = RunningStats.create(dim=dim, device=device, dtype=torch.float32)

        # update with at most remaining samples
        bsz = images.shape[0]
        take = min(bsz, args.max_samples - n_pairs)
        if take <= 0:
            break

        if args.mode == "image":
            stats.update(img_feat[:take])  # type: ignore[index]
        elif args.mode == "text":
            stats.update(txt_feat[:take])  # type: ignore[index]
        else:  # both
            stats.update(img_feat[:take])  # type: ignore[index]
            stats.update(txt_feat[:take])  # type: ignore[index]

        n_pairs += take
        pbar.set_postfix({"pairs": n_pairs})

    if stats is None:
        raise RuntimeError("No samples processed. Check that your dataset provides text annotations and paths are correct.")

    mu, std = stats.finalize(eps=1e-6)
    out = {
        "mode": args.mode,
        "pairs": n_pairs,
        "mu": mu.detach().cpu(),
        "sigma": std.detach().cpu(),
        "feature_dim": mu.numel(),
        "normalized": True,
        "model_name": args.model_name,
        "pretrained_path": args.pretrained_path,
    }
    torch.save(out, args.save_path)
    logger.info(f"Saved stats to {args.save_path}")
    logger.info(f"mode={args.mode}, pairs={n_pairs}, dim={out['feature_dim']}")


if __name__ == "__main__":
    main()

