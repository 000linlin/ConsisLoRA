import os 
from pipeline_demo import StableDiffusionXLPipelineLoraGuidance


def inference(
    pipeline: StableDiffusionXLPipelineLoraGuidance,
    prompt: str,
    content_image_lora_path: str = None,
    style_image_lora_path: str = None,
    add_positive_content_prompt: str = None,
    add_negative_content_prompt: str = None,
    add_positive_style_prompt: str = None,
    add_negative_style_prompt: str = None,
    lora_scaling: list[float] = [1., 1.],
    guidance_scale: float = 7.5,
    content_guidance_scale: float = 0.,
    style_guidance_scale: float = 0.,
    num_images_per_prompt: int = 1,
    num_steps: int = 30, 
    output_dir: str = "inference-image",
    generator=None
):
    
    pipeline.unload_lora_checkpoint()
    pipeline.load_lora_checkpoint(content_image_lora_path, style_image_lora_path)
    
    images = pipeline(
        prompt=prompt, 
        lora_scaling=lora_scaling,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps, 
        add_positive_content_prompt=add_positive_content_prompt,
        add_negative_content_prompt=add_negative_content_prompt,
        add_positive_style_prompt=add_positive_style_prompt,
        add_negative_style_prompt=add_negative_style_prompt,    
        content_guidance_scale=content_guidance_scale,   
        style_guidance_scale=style_guidance_scale, 
        num_images_per_prompt=num_images_per_prompt, 
        generator=generator
    ).images

    os.makedirs(output_dir, exist_ok=True)
    # save
    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f'{i}.jpg'))

    print(f"saved images for {output_dir}")

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="inference script")
    parser.add_argument("--prompt",
        type=str,
        required=True,
        help="Inference prompt."
    )
    parser.add_argument(
        "--content_image_lora_path",
        type=str,
        default=None,
        help="The Path for ConsisLoRA trained on the content image."
    )
    parser.add_argument(
        "--style_image_lora_path",
        type=str,
        default=None,
        help="The Path for ConsisLoRA trained on the style image."
    )
    parser.add_argument(
        "--add_positive_content_prompt",
        type=str,
        default=None,
        help="The positive prompt for content guidance. if content_guidance_scale=0, this prompt will be ignored."
    )
    parser.add_argument(
        "--add_negative_content_prompt",
        type=str,
        default=None,
        help="The negative prompt for content guidance. if content_guidance_scale=0, this prompt will be ignored."
    )
    parser.add_argument(
        "--add_positive_style_prompt",
        type=str,
        default=None,
        help="The positive prompt for style guidance. if style_guidance_scale=0, this prompt will be ignored."
    )
    parser.add_argument(
        "--add_negative_style_prompt",
        type=str,
        default=None,
        help="The negative prompt for style guidance. if style_guidance_scale=0, this prompt will be ignored."
    )
    parser.add_argument(
        "--lora_scaling",
        type=float,                
        nargs="+",               
        default=[1., 1.],        
        help="List of scaling factors for LoRA, i.g., [content_lora_scaling, style_lora_scaling]."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)."
    )
    parser.add_argument(
        "--content_guidance_scale",
        type=float,
        default=0.,
        help="Guidance scale for enhancing content."
    )
    parser.add_argument(
        "--style_guidance_scale",
        type=float,
        default=0.,
        help="Guidance scale for enhancing style."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="number of images per prompt."
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=30,
        help="The number of denoising steps."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference-images",
        help="The path to save the images"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=29, 
        help="A seed for reproducible inference."
    )
    
    args = parser.parse_args()

    generator=torch.manual_seed(args.seed)
    device = "cuda"

    pipeline = StableDiffusionXLPipelineLoraGuidance.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16,
    ).to(device)

    from diffusers import EulerDiscreteScheduler
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing"
    )

    inference(
        pipeline=pipeline,
        prompt=args.prompt,
        content_image_lora_path=args.content_image_lora_path,
        style_image_lora_path=args.style_image_lora_path,
        add_positive_content_prompt=args.add_positive_content_prompt,
        add_negative_content_prompt=args.add_negative_content_prompt,
        add_positive_style_prompt=args.add_positive_style_prompt,
        add_negative_style_prompt=args.add_negative_style_prompt,
        lora_scaling=args.lora_scaling,
        guidance_scale=args.guidance_scale,
        content_guidance_scale=args.content_guidance_scale,
        style_guidance_scale=args.style_guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        num_steps=args.num_steps, 
        output_dir=args.output_dir,
        generator=generator  
    )