import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel

# ---------- CONFIG ----------
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

UNET_LORA_DIR = "pokeforge_lora/unet_lora"
TEXT_LORA_DIR = "pokeforge_lora/text_encoder_lora"

OUTPUT_DIR = "samples_pokeforge"
TRIGGER = "pokeforge"

NUM_STEPS = 30
GUIDANCE = 7.5
DEVICE = "cuda"

NEGATIVE_PROMPT = (
    "text, watermark, photo, furry, real animal, realistic, human, messy lines," \
    "abstract pattern, fractal, texture, wallpaper, leaves, grass texture," \
    "landscape, background focus, low quality, bad anatomy, 3d render"
)

PROMPTS = [
    f"{TRIGGER}, full body, small, creature, clean lineart, cell shading, normal type, cute, white background",
    f"{TRIGGER}, full body, clean lineart, cell shading, fire type, creature, orange, red, sharp, intimidating, sky background",
    f"{TRIGGER}, full body, clean lineart, cell shading, flying type, creature, blue, smooth body, friendly, simple background",
    f"{TRIGGER}, full body, clean lineart, cell shading, creature, green, leafy, elegant, nature background",
    f"{TRIGGER}, full body, clean lineart, cell shading, electric type, creature, yellow, agile, dynamic pose, simple background",
    f"{TRIGGER}, full body, clean lineart, cell shading, dragon type, creature, black and gold, long tail, powerful, dramatic lighting",
    f"{TRIGGER}, full body, clean lineart, cell shading, creature, turtle, blue, calm",
    f"{TRIGGER}, full body, clean lineart, cell shading, creature, dragon type, cute, colorful",
    f"{TRIGGER}, full body, clean lineart, cell shading, creature, black, white, zebra"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading base pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

print("Loading PEFT LoRA into UNet and Text Encoder...")

pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    UNET_LORA_DIR,
)

pipe.text_encoder = PeftModel.from_pretrained(
    pipe.text_encoder,
    TEXT_LORA_DIR,
)

pipe.to(DEVICE)


def generate_image(prompt: str, seed: int, filename: str):
    print(f"→ Generating: {prompt} (seed={seed})")
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    out = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        generator=generator,
        cross_attention_kwargs={"scale": 0.9},
    )
    image = out.images[0]
    image.save(filename)
    print(f"   saved to {filename}")


if __name__ == "__main__":
    for i, prompt in enumerate(PROMPTS):
        seed = 420 + i
        outfile = os.path.join(OUTPUT_DIR, f"pokeforge_{i+1:03d}.png")
        generate_image(prompt, seed, outfile)

    print("Done!")
    
    from PIL import Image
    import math

    images = [Image.open(os.path.join(OUTPUT_DIR, f"pokeforge_{i+1:03d}.png")) for i in range(len(PROMPTS))]
    grid_size = math.ceil(math.sqrt(len(images)))
    img_width, img_height = images[0].size

    grid_img = Image.new('RGB', (img_width * grid_size, img_height * grid_size))

    for idx, img in enumerate(images):
        x = (idx % grid_size) * img_width
        y = (idx // grid_size) * img_height
        grid_img.paste(img, (x, y))

    grid_img.save(os.path.join(OUTPUT_DIR, "grid.png"))
    print(f"Grid image saved to {os.path.join(OUTPUT_DIR, 'grid.png')}")