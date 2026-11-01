"""
Combine batch 1 and batch 2 CF results into a single file for analysis.

Usage:
  python3 combine_cf_batches.py
"""
import json
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

b1_path = "experiments/real_vsr_v3_t0_cf100.json"
b2_path = "experiments/real_vsr_v3_t0_cf_batch2.json"
out_path = "experiments/real_vsr_v3_t0_cf_combined.json"

for p in [b1_path, b2_path]:
    if not os.path.exists(p):
        partial = p + ".partial"
        if os.path.exists(partial):
            print(f"WARNING: using partial {partial}")
        else:
            print(f"Missing: {p}")
            exit(1)

with open(b1_path if os.path.exists(b1_path) else b1_path + ".partial") as f:
    b1 = json.load(f)
with open(b2_path if os.path.exists(b2_path) else b2_path + ".partial") as f:
    b2 = json.load(f)

# Dedup by image_id (batch 1 takes priority if any overlap)
b1_ids = {r["image_id"] for r in b1}
b2_new = [r for r in b2 if r["image_id"] not in b1_ids]

combined = b1 + b2_new
print(f"Batch 1: {len(b1)}, Batch 2 new: {len(b2_new)}, Combined: {len(combined)}")

with_ca = sum(1 for r in combined if r.get("cell_answers"))
print(f"Items with cell_answers: {with_ca}/{len(combined)}")

with open(out_path, "w") as f:
    json.dump(combined, f, indent=2)
print(f"Saved to {out_path}")
