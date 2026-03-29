
INSTANCE_PROMPT="A [c]"
IMAGE_PATH="data_demo/content/modern.jpg"
OUTPUT_DIR="lora-weights/content/modern"

start_x0_loss_steps=500    # Step to begin x0 loss
lr=2e-4                    # Learning rate for noise loss
second_lr=1e-4             # Learning rate for x0 loss

accelerate launch train_consislora.py \
    --instance_prompt="$INSTANCE_PROMPT" \
    --image_path="$IMAGE_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --start_x0_loss_steps="$start_x0_loss_steps" \
    --rank=64 \
    --lr="$lr" \
    --second_lr="$second_lr" \
    --max_train_steps=1500 \
    --checkpointing_steps=500 \
    --resolution=1024 \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --seed=0
    