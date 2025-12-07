from datasets import load_dataset

dataset = load_dataset("svjack/pokemon-blip-captions-en-ja")

import os

os.makedirs("pokemon_like", exist_ok=True)

for idx, item in enumerate(dataset["train"]):
    img = item["image"]
    caption = item["en_text"] or ""
    img.save(f"pokemon_like/{idx:05d}.png")
    with open(f"pokemon_like/{idx:05d}.txt", "w", encoding="utf-8") as f:
        f.write(caption)
