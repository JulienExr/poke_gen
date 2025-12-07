import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel


MODEL_NAME = "runwayml/stable-diffusion-v1-5"
UNET_LORA_PATH = "pokeforge_lora/unet_lora"
TEXT_ENCODER_LORA_PATH = "pokeforge_lora/text_encoder_lora"

TRIGGER_WORD = "pokeforge"

PROMPT = (f"{TRIGGER_WORD}, full body, creature, black, white, dragon type")

NEGATIVE_PROMPT = "lowres, bad anatomy, error body, error arms, error legs, error hands, missing fingers," \
    " extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid," \
    " mutilated, mutation, deformed, blurry, dehydrated, bad proportions, cloned face, disfigured, gross proportions," \
    " malformed limbs, missing limbs, floating limbs, disconnected limbs, blurry, deformed face, lopsided face, bad anatomy, messy lines, extra eyes"

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

pipe.unet = PeftModel.from_pretrained(pipe.unet, UNET_LORA_PATH)
pipe.unet.to(device)
pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, TEXT_ENCODER_LORA_PATH)
pipe.text_encoder.to(device)

with torch.autocast(device):
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=35,
        guidance_scale=7.5,
        height=512,
        width=512,
        cross_attention_kwargs={"scale": 0.7},
    ).images[0]

output_path = "generated_pokeforge.png"
image.save(output_path)
print(f"Image saved to {output_path}")