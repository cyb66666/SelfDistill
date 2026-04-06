# python -m src.training.compute_rafa_prior_stats \
#   --rs3-tar-dir "/root/data/rs3_filtered" \
#   --model-name "ViT-B-32" \
#   --pretrained-path "/root/checkpoint/RS5M_ViT-B-32.pt" \
#   --mode text \
#   --max-samples 5000 \
#   --save-path "./rafa_stats_text.pt" \
#   --batch-size 24 \
#   --num-workers 8

python -m src.training.compute_rafa_prior_stats \
  --rs3-tar-dir "/root/data/rs3_filtered" \
  --model-name "ViT-B-32" \
  --pretrained-path "/root/checkpoint/RS5M_ViT-B-32.pt" \
  --mode image \
  --max-samples 5000 \
  --save-path "./rafa_stats_image.pt" \
  --batch-size 24 \
  --num-workers 8

python -m src.training.compute_rafa_prior_stats \
  --rs3-tar-dir "/root/data/rs3_filtered" \
  --model-name "ViT-B-32" \
  --pretrained-path "/root/checkpoint/RS5M_ViT-B-32.pt" \
  --mode both \
  --max-samples 5000 \
  --save-path "./rafa_stats_both.pt" \
  --batch-size 24 \
  --num-workers 8
