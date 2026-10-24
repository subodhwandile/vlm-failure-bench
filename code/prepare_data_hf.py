"""
Lightweight data prep using HuggingFace datasets.
Downloads only the images we actually need for the pilot — no 20GB zip.

Uses VQAv2 validation split streamed from HuggingFace.
Total download: ~50-100 MB for 200 samples.

Run: python prepare_data_hf.py --n_samples 200
"""

import json
import os
import argparse
import urllib.request
from pathlib import Path

DATA_DIR = Path("./data")
IMAGES_DIR = DATA_DIR / "images"


def download_vqa_samples(n_samples: int = 200, output_qa: str = "data/qa_samples.json"):
    """
    Stream VQAv2 val split from HuggingFace — downloads only the samples we need.
    Focuses on question types where VLMs commonly fail:
      - counting, spatial, attribute, yes/no with spatial reasoning
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Run: pip install datasets")
        raise

    print(f"Streaming VQAv2 val split (this downloads ~500MB index but streams images on demand)...")
    ds = load_dataset("HuggingFace/vqav2-small", split="validation", streaming=True)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Question types where VLMs commonly fail — best for our experiment
    target_types = {"how many", "what color", "what is", "where is", "which", "is there"}

    samples = []
    seen = set()

    for item in ds:
        if len(samples) >= n_samples:
            break

        question = item["question"].lower()
        # Skip trivial yes/no unless spatial
        if item.get("question_type") == "yes/no" and "where" not in question and "left" not in question and "right" not in question:
            continue

        image_id = str(item["image_id"])
        if image_id in seen:
            continue
        seen.add(image_id)

        # Save image
        image_file = f"{image_id}.jpg"
        image_path = IMAGES_DIR / image_file
        if not image_path.exists():
            try:
                item["image"].save(str(image_path))
            except Exception as e:
                print(f"  Could not save image {image_id}: {e}")
                continue

        # Get best answer (most common among annotators)
        answers = item.get("answers", [])
        if not answers:
            continue
        answer_counts = {}
        for a in answers:
            ans = a["answer"].lower().strip()
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
        best_answer = max(answer_counts, key=answer_counts.get)

        samples.append({
            "image_id": image_id,
            "image_file": image_file,
            "question": item["question"],
            "answer": best_answer,
            "question_type": item.get("question_type", "unknown"),
            "answer_confidence": max(answer_counts.values()) / len(answers)
        })

        if len(samples) % 20 == 0:
            print(f"  Collected {len(samples)}/{n_samples} samples...")

    print(f"Collected {len(samples)} samples, saved images to {IMAGES_DIR}")

    with open(output_qa, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"QA file written to {output_qa}")
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output", default="data/qa_samples.json")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    download_vqa_samples(args.n_samples, args.output)
    print("\nDone. Next:")
    print("  python pilot_experiment.py --n_samples 50 --model claude-haiku-5-5")
