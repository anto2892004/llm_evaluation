from datasets import load_dataset
import os

# Set your target folder path
save_dir = "pubmed_dataset"
os.makedirs(save_dir, exist_ok=True)

# Load the dataset with correct config and split
dataset = load_dataset("qiaojin/PubMedQA", name="pqa_labeled", split="train")

# Save as a local CSV file
dataset.to_csv(os.path.join(save_dir, "pqa_labelled.csv"), index=False)

print("Dataset downloaded and saved at:", save_dir)
