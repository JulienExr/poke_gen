import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

from dataset import PokeforgeDataset

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DATASET_DIR = "dataset"
OUTPUT_DIR = "pokeforge_lora"
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
LORA_RANK = 8
GRADIENT_ACCUMULATION_STEPS = 8
MIXED_PRECISION = "fp16"



def main():
    accelerator = Accelerator(mixed_precision=MIXED_PRECISION, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    lora_config = LoraConfig(
        r=LORA_RANK,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
    )

    text_lora_config = LoraConfig(
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        )

    unet = get_peft_model(unet, lora_config)
    text_encoder = get_peft_model(text_encoder, text_lora_config)

    if accelerator.is_main_process:
        print("Trainable UNet parameters:")
        unet.print_trainable_parameters()
        print("Trainable Text Encoder parameters:")
        text_encoder.print_trainable_parameters()

    dataset = PokeforgeDataset(DATASET_DIR, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(list(unet.parameters()) + list(text_encoder.parameters()), lr=LEARNING_RATE)

    weights_dtype = torch.float16 if MIXED_PRECISION == "fp16" else torch.float32

    vae.to(accelerator.device, dtype=weights_dtype)

    unet, text_encoder, optimizer, dataloader = accelerator.prepare(unet, text_encoder, optimizer, dataloader)

    pbar = tqdm(range(NUM_EPOCHS * len(dataloader)), disable=not accelerator.is_main_process)
    pbar.set_description("Training Pokeforge LoRA")

    unet.train()
    text_encoder.train()
    vae.eval()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=weights_dtype)).latent_dist.sample()
                    latents = latents * 0.18215
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                total_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / (step + 1)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Average Loss: {avg_loss:.4f}")

    if accelerator.is_main_process:
        unet_to_save = accelerator.unwrap_model(unet)
        text_encoder_to_save = accelerator.unwrap_model(text_encoder)
        unet_to_save.save_pretrained(os.path.join(OUTPUT_DIR, "unet_lora"))
        text_encoder_to_save.save_pretrained(os.path.join(OUTPUT_DIR, "text_encoder_lora"))
        print("Model saved in", OUTPUT_DIR)
    
if __name__ == "__main__":
    main()