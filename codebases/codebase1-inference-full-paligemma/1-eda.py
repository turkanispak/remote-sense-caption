import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
import sys
from io import StringIO
import logging

# Download NLTK assets
nltk.download('stopwords')

# === Paths ===
DATA_DIR = r"C:\Users\Airlab\Desktop\turkanispak\di725-project\RISCM"
IMAGE_DIR = os.path.join(DATA_DIR, 'resized')
CAPTIONS_CSV = os.path.join(DATA_DIR, "captions.csv")
EDA_DIR = os.path.join(DATA_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

# === Logging Setup ===
log_path = os.path.join(EDA_DIR, "eda_report.txt")
log_stream = StringIO()
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler(log_stream)]
)

# === Load Data ===
captions_df = pd.read_csv(CAPTIONS_CSV)
logging.info(f"Loaded {len(captions_df)} rows from captions.csv")
logging.info(f"\n--- DATASET STRUCTURE ---\n{captions_df.info()}")
logging.info(f"\n--- SPLIT DISTRIBUTION ---\n{captions_df['split'].value_counts()}")

# === Melt caption columns into a single dataframe ===
captions_long_df = captions_df.melt(
    id_vars=["source", "split", "image"],
    value_vars=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"],
    var_name="caption_id",
    value_name="caption"
)


# === Missing Image Check ===
missing_images = []
for img in captions_long_df['image'].unique():
    img_path = os.path.join(IMAGE_DIR, img)
    if not os.path.exists(img_path):
        missing_images.append(img)
logging.info(f"\nMissing image files: {len(missing_images)}")

# === Caption Length Analysis ===
captions_long_df['caption_len'] = captions_long_df['caption'].apply(lambda x: len(str(x).split()))
captions_long_df['char_len'] = captions_long_df['caption'].apply(lambda x: len(str(x)))

# --- Boxplot by split ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=captions_long_df, x='split', y='caption_len')
plt.title("Boxplot of Caption Lengths by Split")
plt.ylabel("Word Count")
plt.savefig(os.path.join(EDA_DIR, "boxplot_caption_length_per_split.png"))
plt.close()

# --- Histogram overall ---
plt.figure(figsize=(10, 6))
sns.histplot(captions_long_df['caption_len'], bins=30, kde=True)
plt.title("Distribution of Caption Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.savefig(os.path.join(EDA_DIR, "hist_caption_word_counts.png"))
plt.close()

# === Token Analysis ===
stop_words = set(stopwords.words('english'))
def clean_tokens(text):
    # Lowercase and remove punctuation manually
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return [w for w in words if w not in stop_words]

captions_long_df['tokens'] = captions_long_df['caption'].apply(clean_tokens)
all_tokens = [token for tokens in captions_long_df['tokens'] for token in tokens]
token_counts = Counter(all_tokens)
top_tokens = token_counts.most_common(30)

# --- Top Tokens Barplot ---
token_df = pd.DataFrame(top_tokens, columns=['token', 'count'])
plt.figure(figsize=(12, 6))
sns.barplot(data=token_df, x='token', y='count')
plt.xticks(rotation=45)
plt.title("Top 30 Most Frequent Tokens (Excluding Stopwords)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "bar_top_tokens.png"))
plt.close()

# === Caption Diversity ===
caption_var_df = captions_long_df.groupby('image')['caption'].apply(lambda x: len(set(x)))
caption_var_df = caption_var_df.reset_index(name='distinct_captions')

# --- Boxplot: Number of Unique Captions per Image ---
plt.figure(figsize=(10, 5))
sns.boxplot(x=caption_var_df['distinct_captions'])
plt.title("Boxplot of Distinct Captions per Image")
plt.xlabel("Number of Unique Captions")
plt.savefig(os.path.join(EDA_DIR, "boxplot_caption_diversity.png"))
plt.close()

# === JSON Summary ===
summary = {
    "total_captions": len(captions_long_df),
    "unique_images": captions_long_df['image'].nunique(),
    "missing_images": len(missing_images),
    "avg_caption_length": captions_long_df['caption_len'].mean(),
    "max_caption_length": captions_long_df['caption_len'].max(),
    "min_caption_length": captions_long_df['caption_len'].min(),
    "top_tokens": top_tokens
}
with open(os.path.join(EDA_DIR, "eda_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

# Save console output explicitly
with open(log_path, 'a') as f:
    f.write("\n--- CONSOLE LOG ---\n")
    f.write(log_stream.getvalue())

print("EDA complete. Outputs saved to:", EDA_DIR)
