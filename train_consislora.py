import diffusers
import transformers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel
)
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection
)

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

import os
import logging
from PIL import Image
from tqdm.auto import tqdm
from peft import LoraConfig, inject_adapter_in_model
from diffusers.utils.torch_utils import is_compiled_module
from utils import unet_lora_state_dict, encode_prompt, filter_consislora


# Global values
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
BLOCKS = {
    "content_lora": "up_blocks.0.attentions.0",
    "style_lora": "up_blocks.0.attentions.1"
}
logger = get_logger(__name__)


# Prepare train image
def load_image(
        image_path: str, 
        resolution: int = None, 
        center_crop: bool = False
):
    
    pil_image = Image.open(image_path).convert('RGB')
    ori_h = tar_h = pil_image.height
    ori_w = tar_w = pil_image.width
    # Prepare add_time_ids for sdxl
    add_time_ids = [ori_h, ori_w, 0, 0,  tar_h, tar_w]
    logger.info(f"  Loaded input image of size (y, x) = ({ori_h}, {ori_w}) from {image_path}")
    if resolution is not None:
        train_resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        pil_image = train_resize(pil_image)
        y = x = 0
        tar_h, tar_w = pil_image.height, pil_image.width
        if center_crop:
            logger.info(f"  Center crop the input image to the resolution")
            train_crop = transforms.CenterCrop(args.resolution)
            pil_image = train_crop(pil_image)
            y = max(0, int(round((ori_h - resolution) / 2.0)))
            x = max(0, int(round((ori_w - resolution) / 2.0)))
            tar_h = tar_w = resolution
        add_time_ids = [ori_h, ori_w, y, x, tar_h, tar_w]

    logger.info(f"  The size of train image is ({tar_h}, {tar_w})")
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]
    )
    image_tensor = train_transforms(pil_image)[None, ...] # add batch dimension
    return image_tensor, add_time_ids


# Prepare lora adpater
def load_lora_adpater(
        unet: UNet2DConditionModel, 
        rank: int, 
        content_lora_path: str = None, 
):

    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    block = unet.get_submodule(BLOCKS["style_lora"])
    inject_adapter_in_model(unet_lora_config, block, "style_lora")
    
    adapter_names = ["style_lora"] # trainable adapter
    if content_lora_path is None:
        block = unet.get_submodule(BLOCKS["content_lora"])
        inject_adapter_in_model(unet_lora_config, block, "content_lora")
        adapter_names.append("content_lora") 
    else:
        content_lora, _ = StableDiffusionXLPipeline.lora_state_dict(content_lora_path)
        content_lora, _ = filter_consislora(content_lora)  
        unet.load_attn_procs(content_lora, adapter_name="content_lora")
        logger.info(f"  Loaded the content lora from {content_lora_path}")

    logger.info(f"  Trainable adapters = {adapter_names}")
    unet.set_adapters(["content_lora", "style_lora"], [1., 1.])

    # Set trainable lora parameters
    for name, param in unet.named_parameters():
        param.requires_grad = False
        if ("lora_A" in name) or ("lora_B" in name):
            # only upcast lora parameters into fp32
            param.data = param.to(torch.float32)
            if any(adapter in name for adapter in adapter_names):
                param.requires_grad = True

    return filter(lambda p: p.requires_grad, unet.parameters())


def train(
    instance_prompt: str,
    image_path: str,
    output_dir: str,
    start_x0_loss_steps: int = None,
    content_lora_path: str = None,  
    max_train_steps: int = 1000,
    rank: int = 64,
    lr: float = 2e-4,
    second_lr: float = 1e-4,
    scale_lr: bool = False,
    resolution: int = 1024,
    center_crop: bool = False,
    checkpointing_steps: int = 500,
    mixed_precision: str = None,
    use_8bit_adam: bool = False,
    noise_offset: float = None,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    seed: int = None 
):
    
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    device=accelerator.device

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    # Prepare all model 
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    tokenizer_one = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_two = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device=device)

    # We only train the additional adapter LoRA layers
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights to half-precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(device, dtype=weight_dtype)
    # The VAE is always in float32 to avoid NaN losses.
    vae.to(device, dtype=torch.float32)

    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)

    # Prepare lora parameters
    lora_parameters = load_lora_adpater(unet, rank, content_lora_path)

    if use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
        logger.info(f"  Using 8bit AdamW optimizer for training")
    else:
        optimizer_class = torch.optim.AdamW

    # Prepare optimizer
    optimizer = optimizer_class(
        lora_parameters,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )
    
    # If not passed in, only the noise loss is used
    if start_x0_loss_steps is None:
        start_x0_loss_steps = max_train_steps

    if second_lr is None:
        second_lr = lr

    if scale_lr:
        lr = (lr * gradient_accumulation_steps * accelerator.num_processes)
        second_lr = (second_lr * gradient_accumulation_steps * accelerator.num_processes)

    def lr_lambda(step):
        if step < start_x0_loss_steps:
            return 1. 
        else:
            return second_lr / lr

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # create custom saving so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    lora_state_dict = unet_lora_state_dict(model, ["content_lora", "style_lora"])
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                
                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=lora_state_dict,
            )

    accelerator.register_save_state_pre_hook(save_model_hook)

    # Prepare everything with `accelerator`.
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    # Prepare train image
    image, add_time_ids = load_image(image_path, resolution, center_crop)
    image = image.to(device, dtype=vae.dtype)
    
    # Prepare train latents
    z_0 = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
    z_0 = z_0.to(weight_dtype)

    # Prepare prompt_embeds and added_cond_kwargs for sdxl
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=[text_encoder_one , text_encoder_two], 
            tokenizers=[tokenizer_one, tokenizer_two], 
            prompt=instance_prompt
        )
        add_time_ids = torch.tensor(
            data=list(add_time_ids), 
            device=device, 
            dtype=weight_dtype
        )
        added_cond_kwargs = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_prompt_embeds
        }
    
    # Running training
    logger.info("***** Running training *****")
    logger.info(f"  Instant prompt = {instance_prompt}")
    logger.info(f"  Max train steps = {max_train_steps}")
    logger.info(f"  Train image path = {image_path}")
    logger.info(f"  Start x0 loss steps = {start_x0_loss_steps}")
    logger.info(f"  Init lr = {lr}")
    logger.info(f"  Second lr = {second_lr}")
    logger.info(f"  Resolution = {resolution}")
    logger.info(f"  Lora rank = {rank}")

    progress_bar = tqdm(range(0, max_train_steps), desc="Steps")
    logs = {"x0_loss": 0., "noise_loss": 0.}
    for step in range(max_train_steps):
        unet.train()

        with accelerator.accumulate(unet):
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(z_0)
            if noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += noise_offset * torch.randn(
                    size=(z_0.shape[0], z_0.shape[1], 1, 1), device=z_0.device,
                )

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (z_0.shape[0],),
                device=z_0.device,
            ).long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            z_t = noise_scheduler.add_noise(z_0, noise, timesteps)

            model_pred = unet(
                z_t,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # noise loss
            if step < start_x0_loss_steps:
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                logs["noise_loss"] = loss.detach().item()
            # x0 loss
            else:
                # z(t) -> predicted z(0)
                z_t = noise_scheduler.step(model_pred, timesteps, z_t).pred_original_sample
                loss = F.mse_loss(z_t.float(), z_0.float(), reduction="mean")
                logs["x0_loss"] = loss.detach().item()

            # backward
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (lora_parameters)
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            logs["lr"] = lr_scheduler.get_last_lr()[0]
            logs["t"] = timesteps.item()
            progress_bar.update()
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process:
            if (step + 1) % checkpointing_steps == 0:
                checkpoint = f"checkpoint-{step + 1}"
                save_path = os.path.join(output_dir, checkpoint) 

                # save lora weight
                accelerator.save_state(save_path)
                logger.info(f"saved {checkpoint} to {save_path}")

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet = unet.to(torch.float32)
        lora_state_dict = unet_lora_state_dict(unet, ["content_lora", "style_lora"])

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=lora_state_dict,
        )

    accelerator.end_training()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="The prompt with identifier specifying the instance.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="The path of training image."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--start_x0_loss_steps",
        type=int,
        default=None,
        help="The number of steps to turn on x0 loss."
    )
    parser.add_argument(
        "--content_lora_path",
        type=str,
        default=None,
        help=(
            "The path of pre-trained Content LoRA is passed in. If passed in, this LoRA weights will be frozen"
            " and only the Style LoRA will be trained."
        )
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Initial learning rate (When using noise loss) to use."
    )
    parser.add_argument(
        "--second_lr",
        type=float,
        default=None,
        help="This learning rate is enabled when using x0 loss. If not set, the initial learning rate is used."
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs and gradient accumulation steps.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1500,
        help="Total number of training steps to perform."
    )
    parser.add_argument(
        "--max_grad_norm", 
        type=float, 
        default=1.0, 
        help="Max gradient norm."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the short side of images will be"
            " resized to the resolution."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        default=False,
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--noise_offset", 
        type=float, 
        default=None, 
        help="The scale of noise offset."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )

    args = parser.parse_args()

    train(
        instance_prompt=args.instance_prompt,
        image_path=args.image_path,
        output_dir=args.output_dir,
        start_x0_loss_steps=args.start_x0_loss_steps,
        content_lora_path=args.content_lora_path,
        max_train_steps=args.max_train_steps, 
        rank=args.rank,
        lr=args.lr, 
        second_lr=args.second_lr,
        scale_lr=args.scale_lr,
        resolution=args.resolution,
        center_crop=args.center_crop,
        checkpointing_steps=args.checkpointing_steps,
        mixed_precision=args.mixed_precision,
        use_8bit_adam=args.use_8bit_adam,
        noise_offset=args.noise_offset,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed
    )