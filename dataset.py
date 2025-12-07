import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PokeforgeDataset(Dataset):
    def __init__(self, root_dir, tokenizer, size=(512, 512)):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.size = size
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(Path(root_dir).glob(f"*{ext}")))

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption_path = img_path.with_suffix('.txt')
        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        else:
            caption = "pokeforge, creature, full body, high detail, cute"
        tokenized_caption = self.tokenizer(
            caption,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt'
        ).input_ids.squeeze(0)

        return {
            'pixel_values': image,
            'input_ids': tokenized_caption
        }
    
if __name__ == "__main__":
    from transformers import CLIPTokenizer
    
    print("testing PokeforgeDataset...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        ds = PokeforgeDataset("dataset", tokenizer)
        
        if len(ds) > 0:
            item = ds[0]
            print(f"Image shape: {item['pixel_values'].shape}") 
            print(f"Token IDs: {item['input_ids'][:10]}...")
            print("All good!")
        else:
            print("The dataset folder is empty or not found.")
            
    except Exception as e:
        print(f"Error: {e}")