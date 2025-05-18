"""
################################################################################
# DEBUG-DATASET.PY - REMOTE SENSING CAPTIONING PROJECT
#
# This script was created to deeply verify the correctness and usability
# of the RISCM dataset (resized images + captions) before fine-tuning
# the PaliGemma-3B-Mix-224 model with LoRA.
#
# STEP-BY-STEP WHAT WAS CHECKED AND FOUND OUT:
#
# 1. Confirmed that the number of available images on disk matched exactly
#        the number of unique image filenames referenced in captions.csv.
#    - 44,521 images present
#    - 44,521 images referenced
#    - No missing images, no extra unreferenced images.
#
# 2. Melted the captions.csv into long format (each image with 5 captions),
#        resulting in 222,605 rows (image-caption pairs).
#    - No melted rows referenced missing images.
#    - No empty or missing captions found.
#
# 3. Successfully loaded a single image + its caption manually.
#    - Confirmed that AutoProcessor correctly processed the image + text.
#    - Processing time ~18 milliseconds per sample.
#
# 4. Issue Encountered:
#    - When trying to use a DataLoader with num_workers=4 (multiprocessing),
#      the loading process froze indefinitely after "Trying to load one batch..."
#    - CPU usage was very low (2%), no heavy I/O, meaning a multiprocessing deadlock.
#
# 5. Root Cause:
#    - On Windows, PyTorch uses 'spawn' multiprocessing.
#    - Complex objects like AutoProcessor inside Dataset cause process fork issues.
#    - This is a common problem for large processors and heavy datasets on Windows.
#
# 6. Solution:
#    - Set num_workers=0 in DataLoader to disable multiprocessing.
#    - Loading then worked instantly, and a full batch was retrieved.
#    - Dataset proven to be usable without crashing.
#
# 7. Final Status:
#    - Dataset integrity: Verified OK.
#    - Single image load: OK.
#    - Full batch load: OK with num_workers=0.
#
################################################################################
"""

import pandas as pd
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
import torch
import time

# === 1. PATHS ===
DATA_DIR = r"C:\Users\Airlab\Desktop\turkanispak\di725-project\RISCM"
IMAGE_DIR = os.path.join(DATA_DIR, "resized")
CAPTIONS_CSV = os.path.join(DATA_DIR, "captions.csv")

# === 2. LOAD ===
df = pd.read_csv(CAPTIONS_CSV)

available_images = set(os.listdir(IMAGE_DIR))
print(f"Available images in folder: {len(available_images):,}")

unique_images_in_csv = set(df['image'])
print(f"Unique images referenced in captions.csv: {len(unique_images_in_csv):,}")

missing_images = unique_images_in_csv - available_images
extra_images = available_images - unique_images_in_csv

print("\nAnalysis:")
print(f"- Images referenced in CSV but missing on disk: {len(missing_images):,}")
print(f"- Images present on disk but not referenced in CSV: {len(extra_images):,}")

if missing_images:
    print("There are missing images. Dataset is incomplete.")
else:
    print("No missing images. All referenced images are available.")

# === 3. VERIFY FULL DATASET CONSTRUCTION ===
melted = df.melt(id_vars=["source", "split", "image"],
                 value_vars=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
                 var_name="caption_id", value_name="caption")

print(f"\nAfter melting, total rows (image-caption pairs): {len(melted):,}")

invalid_rows = melted[~melted['image'].isin(available_images)]
if not invalid_rows.empty:
    print(f"{len(invalid_rows):,} melted rows reference missing images.")
else:
    print("All melted rows reference existing images.")

empty_captions = melted[melted['caption'].isnull() | (melted['caption'].str.strip() == "")]
print(f"Melted rows with empty captions: {len(empty_captions):,}")

if len(empty_captions) > 0:
    print("Warning: Some captions are missing text.")
else:
    print("All captions are non-empty.")

print("\nDataset construction analysis complete. No files modified.")

# === 4. LOAD PROCESSOR ===
model_name = "google/paligemma-3b-mix-224"
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

# === 5. MANUAL SINGLE IMAGE LOAD AND TEST ===
print("\nTesting manual load of a single image + caption...")

sample = melted.iloc[0]
image_path = os.path.join(IMAGE_DIR, sample['image'])
caption = sample['caption']

assert os.path.exists(image_path), f"Image does not exist: {image_path}"

# Load image
image = Image.open(image_path).convert("RGB")
print(f"Loaded image: {sample['image']}")
print(f"Caption: {caption}")

# Process
start_time = time.time()
inputs = processor(
    images=image,
    text="<image> " + caption, 
    return_tensors="pt",
    padding="max_length",
    max_length=50,
    truncation=True
)
end_time = time.time()

print("\nProcessor output:")
for k, v in inputs.items():
    print(f"{k}: shape = {v.shape}, dtype = {v.dtype}")
print(f"Processing time: {(end_time - start_time):.3f} seconds")

# === 6. CONSTRUCT DATASET + DATALOADER ===
print("\nConstructing Dataset and DataLoader for test...")

class RSICDDebugDataset(Dataset):
    def __init__(self, dataframe, processor, image_dir, max_length=50):
        self.dataframe = dataframe.reset_index(drop=True)
        self.processor = processor
        self.image_dir = image_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image'])
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(
            images=image,
            text="<image> " + row['caption'],  # Add <image> prefix here to match expectations
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

batch_size = 8

dataset = RSICDDebugDataset(melted, processor, IMAGE_DIR, max_length=50)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(f"Dataset size: {len(dataset):,}")
print(f"Trying to load one batch with batch_size={batch_size}...")

# Try loading one batch
batch = next(iter(loader))

print("\nBatch loaded successfully.")
for k, v in batch.items():
    print(f"{k}: shape = {v.shape}, dtype = {v.dtype}")
