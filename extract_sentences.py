import csv
import random
import os

INPUT_FILE = "data/raw/english_language_learning_1M.csv"
OUTPUT_FILE = "data/cleaned_sentences.txt"
SAMPLE_SIZE = 500

def extract_sentences():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    sentences = []
    
    # Read CSV efficiently
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "correct_text" in row and row["correct_text"]:
                sentences.append(row["correct_text"])
    
    unique_sentences = list(set(sentences))
    print(f"Found {len(unique_sentences)} unique sentences.")
    
    selected = random.sample(unique_sentences, min(SAMPLE_SIZE, len(unique_sentences)))
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in selected:
            f.write(s + "\n")
            
    print(f"Saved {len(selected)} sentences to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_sentences()
