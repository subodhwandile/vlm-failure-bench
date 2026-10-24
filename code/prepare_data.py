"""
Data preparation: download a small VQA subset for the pilot.

Uses the GQA dataset (balanced val split) — freely available.
Downloads 200 samples with spatial/attribute questions where VLMs commonly fail.

Run: python prepare_data.py
"""

import json
import os
import urllib.request
from pathlib import Path

DATA_DIR = Path("./data")
IMAGES_DIR = DATA_DIR / "images"


def download_gqa_questions(output_path: Path, n: int = 200):
    """
    GQA balanced val split is available at:
    https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip

    For the pilot we use a curated small subset focusing on:
    - Spatial relation questions (left/right/above/below)
    - Attribute questions (color, size, shape)
    - Counting questions
    These are question types where VLMs commonly fail.
    """
    print("NOTE: GQA requires manual download from https://cs.stanford.edu/people/dorarad/gqa/download.html")
    print("Download 'val_balanced_questions.json' and place in ./data/")
    print("Download images from 'images.zip' and extract to ./data/images/")
    print()
    print("Alternatively, use VQAv2 val split from https://visualqa.org/download.html")
    print()

    # Create a small synthetic sample for testing the pipeline
    sample_data = [
        {
            "image_id": "sample_001",
            "image_file": "sample_001.jpg",
            "question": "What color is the object on the left?",
            "answer": "red",
            "question_type": "attribute"
        },
        {
            "image_id": "sample_002",
            "image_file": "sample_002.jpg",
            "question": "How many objects are on the table?",
            "answer": "3",
            "question_type": "counting"
        }
    ]

    with open(output_path, "w") as f:
        json.dump(sample_data, f, indent=2)
    print(f"Created placeholder QA file at {output_path}")
    print("Replace with real GQA or VQAv2 data before running pilot.")


def convert_gqa_to_pilot_format(gqa_questions_path: str, images_dir: str, output_path: str, n: int = 200):
    """
    Convert GQA val_balanced_questions.json to pilot format.
    Focus on question types where spatial/attribute understanding matters.
    """
    with open(gqa_questions_path) as f:
        questions = json.load(f)

    target_types = {"verify", "query"}  # GQA question types
    target_semantic = {"relation", "attribute", "global"}

    samples = []
    for qid, qdata in questions.items():
        if len(samples) >= n:
            break

        image_file = f"{qdata['imageId']}.jpg"
        image_path = os.path.join(images_dir, image_file)

        if not os.path.exists(image_path):
            continue

        samples.append({
            "image_id": qid,
            "image_file": image_file,
            "question": qdata["question"],
            "answer": qdata["answer"],
            "question_type": qdata.get("types", {}).get("semantic", "unknown"),
            "detailed_type": qdata.get("types", {}).get("detailed", "unknown")
        })

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Converted {len(samples)} GQA questions to {output_path}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    qa_output = DATA_DIR / "qa_samples.json"

    gqa_path = DATA_DIR / "val_balanced_questions.json"
    if gqa_path.exists():
        print("Found GQA questions file — converting...")
        convert_gqa_to_pilot_format(str(gqa_path), str(IMAGES_DIR), str(qa_output), n=200)
    else:
        print("GQA not found — creating placeholder...")
        download_gqa_questions(qa_output)
