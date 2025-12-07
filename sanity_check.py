import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image


MODEL_NAME = "runwayml/stable-diffusion-v1-5"
UNET_LORA_PATH = "pokeforge_lora/unet_lora"
TEXT_ENCODER_LORA_PATH = "pokeforge_lora/text_encoder_lora"
SEED = 42
TRIGGER_WORD = "pokeforge"
PROMPT = (f"{TRIGGER_WORD}, creature, full body")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe_base = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(device)

generator = torch.Generator(device).manual_seed(SEED)

image_base = pipe_base(
    prompt=PROMPT,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    generator=generator,
    safety_checker=None,
).images[0]

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

pipe.unet = PeftModel.from_pretrained(pipe.unet, UNET_LORA_PATH)
pipe.unet.to(device, dtype=torch.float16)

pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, TEXT_ENCODER_LORA_PATH)
pipe.text_encoder.to(device, dtype=torch.float16)


generator = torch.Generator(device).manual_seed(SEED)

image_lora = pipe(
    prompt=PROMPT,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
    generator=generator,
).images[0]

w, h = image_base.size
new_image = Image.new('RGB', (w * 2, h))
new_image.paste(image_base, (0, 0))
new_image.paste(image_lora, (w, 0))

comparison_path = "pics/comparison_pokeforge.png"
new_image.save(comparison_path)
print(f"Comparison image saved to {comparison_path}")