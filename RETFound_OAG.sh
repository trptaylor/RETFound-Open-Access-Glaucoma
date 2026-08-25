# ==== Model/settings ====
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
ADAPTATION="finetune"
NUM_CLASS=2

# ==== Data/settings ====
DATA_PATH="/home/tom/Project/Tom_RETFound/rfoagtrial/RETFound-Open-Access-Glaucoma/cropped"
TASK="RETFound_OAG"

# ==== Checkpoint ====
CKPT="/home/tom/Project/Tom_RETFound/rfoagtrial/RETFound-Open-Access-Glaucoma/RETFound.OAG.pth"

# ==== Prediction run ====
python -m torch.distributed.run --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model "$MODEL" \
  --model_arch "$MODEL_ARCH" \
  --global_pool \
  --batch_size 128 \
  --world_size 1 \
  --nb_classes "$NUM_CLASS" \
  --data_path "$DATA_PATH" \
  --input_size 224 \
  --task "$TASK" \
  --adaptation "$ADAPTATION" \
  --eval \
  --predict \
  --resume "$CKPT"
