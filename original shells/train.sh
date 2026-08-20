# ==== Model settings ====
# adaptation {finetune,lp}
ADAPTATION="finetune"
MODEL="RETFound_dinov2"
MODEL_ARCH="retfound_dinov2"
FINETUNE="RETFound_dinov2_meh"

# ==== Data settings ====
# change the dataset name and corresponding class number
DATASET="UKB_quality"
NUM_CLASS=2

# =======================
DATA_PATH="/home/tom/Documents/UKB Quality/fine tuning images"
TASK="retfound_dinov2_UKB_quality_finetune"

torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model "RETFound_dinov2" \
  --model_arch "retfound_dinov2" \
  --finetune "RETFound_dinov2_meh" \
  --savemodel \
  --global_pool \
  --batch_size 12 \
  --world_size 1 \
  --epochs 50 \
  --nb_classes "2" \
  --data_path "/home/tom/Documents/UKB Quality/fine tuning images" \
  --input_size 224 \
  --task "retfound_dinov2_UKB_quality_finetune" \
  --adaptation "finetune"
