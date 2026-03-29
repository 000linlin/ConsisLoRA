
INSTANCE_PROMPT="A [v]"
IMAGE_PATH="data_demo/style/pig.jpg"
OUTPUT_DIR="lora-weights/style/pig"

MAX_TRAIN_STEPS=1500 # Number of training steps for the content lora of style images

# Check if the output directory already exists
if [ ! -d "$OUTPUT_DIR/checkpoint-$MAX_TRAIN_STEPS" ]; then
    # train the content lora of the style image
    accelerate launch train_consislora.py \
        --instance_prompt="$INSTANCE_PROMPT" \
        --image_path="$IMAGE_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --start_x0_loss_steps=500 \
        --rank=64 \
        --lr=2e-4 \
        --second_lr=1e-4 \
        --max_train_steps="$MAX_TRAIN_STEPS" \
        --checkpointing_steps=500 \
        --resolution=1024 \
        --mixed_precision="fp16" \
        --use_8bit_adam \
        --seed=0
fi

STYLE_PROMPT="An image in the style of [v]"
CONTENT_LORA_PATH="$OUTPUT_DIR"
OUTPUT_DIR="${OUTPUT_DIR//style/style_retrain}" # New save directory for style lora 

# train style lora
accelerate launch train_consislora.py \
    --instance_prompt="$STYLE_PROMPT" \
    --image_path="$IMAGE_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --content_lora_path="$CONTENT_LORA_PATH" \
    --start_x0_loss_steps=0 \
    --rank=64 \
    --lr=1e-4 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --resolution=1024 \
    --noise_offset=0.03 \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --seed=0