
# ==== Model/settings ====
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
ADAPTATION="finetune"
NUM_CLASS=2

# ==== Data/settings ====
DATA_PATH="/path/to/your/image/folder"
TASK="RETFound GQ"

# ==== Checkpoint ====
CKPT="/path/to/your/folder/RETFound_OAG/checkpoints/RETFound GQ/RETFound GQ.pth"

# ==== Prediction run ====
torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
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
