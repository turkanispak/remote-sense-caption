# baseline_inference.py

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import evaluate
import wandb
from transformers import AutoProcessor, AutoModelForVision2Seq
import nltk
import torch
import torch.cuda

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# === 1. SETUP ===
nltk.download('punkt')

# Directories
PROJECT_DIR = r"C:\Users\Airlab\Desktop\turkanispak\di725-project"
DATA_DIR = os.path.join(PROJECT_DIR, "RISCM")
IMAGE_DIR = os.path.join(DATA_DIR, 'resized')
CAPTIONS_CSV = os.path.join(DATA_DIR, "captions.csv")

RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "baseline")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Auto Batch Size ===
num_gpus = torch.cuda.device_count()
total_memory_per_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # memory per GPU in GB

if total_memory_per_gpu >= 15:
    base_batch_size = 8
elif total_memory_per_gpu >= 10:
    base_batch_size = 4
else:
    base_batch_size = 2

BATCH_SIZE = base_batch_size * num_gpus

print(f"Auto-selected BATCH_SIZE = {BATCH_SIZE} ({base_batch_size} per GPU × {num_gpus} GPUs, {total_memory_per_gpu:.1f} GB each)")



# === 2. INIT WANDB ===
wandb.init(
    project="remote-sense-caption",   # <-- existing project name
    entity="turkanispak-middle-east-technical-university",  # <-- username/entity
    name="baseline-inference-paligemma",
    config={
        "model": "google/paligemma-3b-mix-224",
        "inference_mode": "zero-shot",
        "device": device
    }
)


# === 3. LOAD PRETRAINED MODEL ===
model_name = "google/paligemma-3b-mix-224"
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# === MULTI-GPU support ===
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel...")
    model = torch.nn.DataParallel(model)


# === 4. LOAD TEST DATA ===
print("Loading RSICD captions...")
df = pd.read_csv(CAPTIONS_CSV)
test_df = df[df['split'] == 'test']

# Melt captions
test_long = test_df.melt(
    id_vars=["source", "split", "image"],
    value_vars=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
    var_name="caption_id",
    value_name="caption"
)

print(f"Total test (image, caption) pairs: {len(test_long)}")

# === 5. INFERENCE ===
predictions = []
references = []
image_names = []

print("Running batched inference...")
# --- BATCHED INFERENCE ---
batch_images = []
batch_texts = []
batch_references = []
batch_image_names = []

for idx, row in tqdm(test_long.iterrows(), total=len(test_long)):
    image_path = os.path.join(IMAGE_DIR, row['image'])

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue

    batch_images.append(image)
    batch_texts.append("<image> caption:")
    batch_references.append(row['caption'])
    batch_image_names.append(row['image'])

    # Process batch
    if len(batch_images) == BATCH_SIZE or idx == len(test_long) - 1:
        inputs = processor(
            images=batch_images,
            text=batch_texts,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: (v.half().cuda() if v.dtype == torch.float32 else v.cuda()) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.module.generate(**inputs, max_new_tokens=50)

        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        predictions.extend([gt.strip() for gt in generated_texts])
        references.extend(batch_references)
        image_names.extend(batch_image_names)

        # Clear batch
        batch_images = []
        batch_texts = []
        batch_references = []
        batch_image_names = []


print(f"Generated {len(predictions)} captions.")

# === 6. SAVE RESULTS ===
result_df = pd.DataFrame({
    "image": image_names,
    "reference_caption": references,
    "predicted_caption": predictions
})
result_csv_path = os.path.join(RESULTS_DIR, "baseline_predictions.csv")
result_df.to_csv(result_csv_path, index=False)
print(f"Saved predictions to {result_csv_path}")

# === 7. EVALUATE BASELINE PERFORMANCE ===
print("Evaluating baseline...")

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

references_list = [[ref] for ref in references]

bleu_score = bleu.compute(predictions=predictions, references=references_list)
meteor_score = meteor.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)

metrics = {
    "BLEU-4": bleu_score['bleu'],
    "METEOR": meteor_score['meteor'],
    "ROUGE-L": rouge_score['rougeL']
}

metrics_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
pd.Series(metrics).to_json(metrics_path, indent=4)
print(f"Saved evaluation metrics to {metrics_path}")

# === 8. LOG TO WANDB ===
wandb.log({
    "BLEU-4": metrics["BLEU-4"],
    "METEOR": metrics["METEOR"],
    "ROUGE-L": metrics["ROUGE-L"]
})

wandb.finish()

print("\n=== BASELINE SCORES ===")
for metric, score in metrics.items():
    print(f"{metric}: {score:.4f}")

print("\nBaseline inference complete!")
