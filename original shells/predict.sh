# ==== Model/settings ====
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
ADAPTATION="finetune"
NUM_CLASS=2

# ==== Data/settings ====
DATA_PATH="/home/tom/Project/Tom_RETFound/EPIC_images_15.12.25/RETFound Predict Practice"
TASK="RF_Predict_Practice"

# ==== Checkpoint ====
CKPT="/home/tom/Project/Tom_RETFound/RETFound/output_dir/retfound_dinov2_all_diagnosis_cropped_10_finetune/checkpoint-best.pth"

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
