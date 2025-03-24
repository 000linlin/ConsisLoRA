from PIL import Image
import torch
from torchvision import transforms


BLOCKS = {
    'content': ['unet.up_blocks.0.attentions.0'],
    'style': ['unet.up_blocks.0.attentions.1'],
}

def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')


def filter_consislora(state_dict):
    try:
        content_lora = {
            k: v for k, v in state_dict.items() if is_belong_to_blocks(k, BLOCKS["content"])
        }
        style_lora = {
            k: v for k, v in state_dict.items() if is_belong_to_blocks(k, BLOCKS["style"])
        }
        return content_lora, style_lora
    except Exception as e:
        raise type(e)(f'failed to filter_lora, due to: {e}')


def load_pil_image(img_path: str, target_size: int = None):
    pil_img = Image.open(img_path).convert('RGB')
    print(f"Loaded input image of size ({pil_img.size}) from {img_path}")
    if target_size is not None:
        resize = transforms.Resize(
            size=target_size, 
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        pil_img = resize(pil_img)
    return pil_img


def unet_lora_state_dict(unet, adapter_name):
    lora_state_dict = {}

    for name, module in unet.named_modules():
        for adapter in adapter_name:
            if hasattr(module, adapter):
                lora_layer = getattr(module, adapter)
                if lora_layer is not None:
                    current_lora_sd = lora_layer.state_dict()
                    if "lora_A" in name: name = name.replace("lora_A", "lora.down")
                    elif "lora_B" in name: name = name.replace("lora_B", "lora.up")
                    for matrix_name, param in current_lora_sd.items():
                        lora_state_dict[f"{name}.{matrix_name}"] = param

    return lora_state_dict


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds
