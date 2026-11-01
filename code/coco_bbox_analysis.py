"""
COCO bounding-box per-instance analysis for FailWhere VSR results.

Scores whether the model's neutral REGION tag hits the actual evidence cells
(subject bbox cell, object bbox cell) — reported separately and as hit_either,
split by correct vs. failed answers.

Three baselines are computed:
  1. Uniform: 1/9 per single entity, 1-(8/9)^2 ≈ 21% for either
  2. Permuted-tag null: shuffle neutral_tagged_region across all items, recompute
  3. Always-center: what accuracy a model that always says "center" achieves

Category matching rules (no substring; explicit plural normalization only):
  exact match → strip trailing 's' → strip trailing 'es' → unmatched
  Multi-word categories matched via word-split overlap with the phrase.

Usage:
  python3 coco_bbox_analysis.py \\
    --results experiments/real_vsr_v3_t0.json \\
    --coco    data/coco/instances_val2017.json
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict, Counter
from pathlib import Path


GRID_CELLS = [
    ("top-left",      0,     0,     1/3,   1/3),
    ("top-center",    1/3,   0,     2/3,   1/3),
    ("top-right",     2/3,   0,     1.0,   1/3),
    ("middle-left",   0,     1/3,   1/3,   2/3),
    ("center",        1/3,   1/3,   2/3,   2/3),
    ("middle-right",  2/3,   1/3,   1.0,   2/3),
    ("bottom-left",   0,     2/3,   1/3,   1.0),
    ("bottom-center", 1/3,   2/3,   2/3,   1.0),
    ("bottom-right",  2/3,   2/3,   1.0,   1.0),
]
N_CELLS = len(GRID_CELLS)


def bbox_center_to_cell(cx_frac: float, cy_frac: float) -> str:
    for name, x0, y0, x1, y1 in GRID_CELLS:
        if x0 <= cx_frac < x1 and y0 <= cy_frac < y1:
            return name
    return "center"  # edge=1.0 fallback


def parse_caption(caption: str):
    """
    'The X is <relation> the Y.' → (subj_phrase, obj_phrase) lowercased.
    Returns (None, None) if pattern doesn't match.
    """
    m = re.match(
        r'the\s+(.+?)\s+is\s+.+?\s+the\s+(.+?)\.?\s*$',
        caption.strip(), re.IGNORECASE
    )
    if m:
        return m.group(1).lower().strip(), m.group(2).lower().strip()
    return None, None


def build_coco_index(coco_path: str):
    """
    Returns:
      cat_names: set of COCO category name strings (lowercase)
      img_anns:  {image_id: [(cat_name, cx_frac, cy_frac), ...]}
    """
    print(f"Loading COCO annotations from {coco_path} ...")
    with open(coco_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"].lower() for c in coco["categories"]}
    cat_names = set(cat_id_to_name.values())

    img_sizes = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        w, h = img_sizes.get(iid, (1, 1))
        x, y, bw, bh = ann["bbox"]
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        cat_name = cat_id_to_name.get(ann["category_id"], "")
        img_anns[iid].append((cat_name, cx, cy))

    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations, "
          f"{len(cat_names)} categories")
    return cat_names, img_anns


def phrase_to_cat(phrase: str, cat_names: set) -> str | None:
    """
    Map a caption noun phrase to a COCO category name.
    Rules (no substring; prevents 'car' → 'carrot', 'cat' → 'scattered'):
      1. Exact match
      2. Strip trailing 's' (dogs→dog)
      3. Strip trailing 'es' (boxes→box) — only if result ≥3 chars
      4. Word-level overlap for multi-word COCO names (e.g. 'dining table')
    """
    phrase = phrase.lower().strip()
    if phrase in cat_names:
        return phrase
    # strip trailing 's'
    if phrase.endswith("s") and phrase[:-1] in cat_names:
        return phrase[:-1]
    # strip trailing 'es'
    if phrase.endswith("es") and len(phrase) > 4 and phrase[:-2] in cat_names:
        return phrase[:-2]
    # multi-word overlap: split phrase into words; check if any cat_name's words
    # are a subset of the phrase's words (whole words only)
    phrase_words = set(phrase.split())
    for cat in cat_names:
        cat_words = set(cat.split())
        if len(cat_words) > 1 and cat_words.issubset(phrase_words):
            return cat
    return None


def evidence_cells_for(image_id: int, cat_name: str,
                        img_anns: dict) -> set[str]:
    """
    Return the SET of 3×3 cells occupied by any annotation of `cat_name`
    in this image.  Empty set if no match.
    """
    cells = set()
    for c, cx, cy in img_anns.get(image_id, []):
        if c == cat_name:
            cells.add(bbox_center_to_cell(cx, cy))
    return cells


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0, center - margin), min(1, center + margin)


def hit_stat(items: list, key: str) -> tuple[int, int, float, float, float]:
    """hits, n, rate, ci_lo, ci_hi"""
    hits = sum(1 for r in items if r.get(key))
    n = len(items)
    lo, hi = wilson_ci(hits, n)
    return hits, n, hits / n if n else 0, lo, hi


def permuted_null(rows: list, n_perm: int = 1000) -> dict:
    """
    Shuffle neutral_tagged_region across all rows n_perm times;
    return mean hit rates for subj/obj/either on correct vs failed.
    """
    tags = [r["neutral_tag"] for r in rows if r.get("neutral_tag")]
    keys = ["hit_subj", "hit_obj", "hit_either"]
    acc = {k: {"correct": [], "fail": []} for k in keys}
    correct_rows = [r for r in rows if not r["is_wrong"] and r.get("neutral_tag") and r.get("neutral_tag") is not None]
    fail_rows    = [r for r in rows if r["is_wrong"]     and r.get("neutral_tag") and r.get("neutral_tag") is not None]

    for _ in range(n_perm):
        shuffled = tags.copy()
        random.shuffle(shuffled)
        tag_map = {r["image_id"]: t for r, t in
                   zip([r for r in rows if r.get("neutral_tag")], shuffled)}
        for subset, name in [(correct_rows, "correct"), (fail_rows, "fail")]:
            sh = sum(1 for r in subset if tag_map.get(r["image_id"]) in r["subj_cells"])
            so = sum(1 for r in subset if tag_map.get(r["image_id"]) in r["obj_cells"])
            se = sum(1 for r in subset if (
                tag_map.get(r["image_id"]) in r["subj_cells"] or
                tag_map.get(r["image_id"]) in r["obj_cells"]))
            n = len(subset)
            acc["hit_subj"][name].append(sh / n if n else 0)
            acc["hit_obj"][name].append(so / n if n else 0)
            acc["hit_either"][name].append(se / n if n else 0)

    return {
        k: {g: sum(v) / len(v) if v else 0 for g, v in gv.items()}
        for k, gv in acc.items()
    }


def main():
    os.chdir(Path(__file__).parent)
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/real_vsr_v3_t0.json")
    parser.add_argument("--coco",    default="data/coco/instances_val2017.json")
    parser.add_argument("--n_perm",  type=int, default=1000)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} results from {args.results}")

    cat_names, img_anns = build_coco_index(args.coco)

    # ── Build per-row evidence cells ──
    rows = []
    parse_fail = 0
    subj_miss = obj_miss = 0

    for r in results:
        image_id    = int(r["image_id"])
        caption     = r["caption"]
        is_wrong    = r["is_wrong"]
        neutral_tag = r.get("neutral_tagged_region")

        subj_phrase, obj_phrase = parse_caption(caption)
        if subj_phrase is None:
            parse_fail += 1

        subj_cat = phrase_to_cat(subj_phrase, cat_names) if subj_phrase else None
        obj_cat  = phrase_to_cat(obj_phrase,  cat_names) if obj_phrase  else None

        subj_cells = evidence_cells_for(image_id, subj_cat, img_anns) if subj_cat else set()
        obj_cells  = evidence_cells_for(image_id, obj_cat,  img_anns) if obj_cat  else set()

        if not subj_cells: subj_miss += 1
        if not obj_cells:  obj_miss  += 1

        hit_subj   = (neutral_tag in subj_cells)  if (neutral_tag and subj_cells) else None
        hit_obj    = (neutral_tag in obj_cells)   if (neutral_tag and obj_cells)  else None
        hit_either = (neutral_tag in (subj_cells | obj_cells)) \
                     if (neutral_tag and (subj_cells or obj_cells)) else None

        rows.append({
            "image_id":   image_id,
            "caption":    caption,
            "is_wrong":   is_wrong,
            "subj":       subj_phrase,
            "obj":        obj_phrase,
            "subj_cat":   subj_cat,
            "obj_cat":    obj_cat,
            "subj_cells": list(subj_cells),
            "obj_cells":  list(obj_cells),
            "neutral_tag": neutral_tag,
            "hit_subj":    hit_subj,
            "hit_obj":     hit_obj,
            "hit_either":  hit_either,
        })

    n_total     = len(rows)
    n_parsed    = n_total - parse_fail
    n_scorable  = sum(1 for r in rows if r["hit_either"] is not None)
    print(f"\nCaption parsed:        {n_parsed}/{n_total} ({100*n_parsed/n_total:.1f}%)")
    print(f"Subject bbox found:    {n_total-subj_miss}/{n_total} ({100*(n_total-subj_miss)/n_total:.1f}%)")
    print(f"Object bbox found:     {n_total-obj_miss}/{n_total} ({100*(n_total-obj_miss)/n_total:.1f}%)")
    print(f"Scorable (hit_either defined): {n_scorable}/{n_total}")

    # ── Subsets ──
    correct_all  = [r for r in rows if not r["is_wrong"]]
    fail_all     = [r for r in rows if r["is_wrong"]]
    correct_sc   = [r for r in correct_all if r["hit_either"] is not None]
    fail_sc      = [r for r in fail_all     if r["hit_either"] is not None]

    # ── Main results table ──
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("PER-INSTANCE NEUTRAL TAG ACCURACY  (neutral prompt, temp=0)")
    print(SEP)
    print(f"{'Metric':<30} {'Correct':>18} {'Failed':>18} {'Delta':>8}")
    print("-" * 70)

    for key, label in [("hit_subj", "hit subject cell"),
                        ("hit_obj",  "hit object cell"),
                        ("hit_either", "hit either cell")]:
        ch, cn, cp, clo, chi = hit_stat(correct_sc, key)
        fh, fn, fp, flo, fhi = hit_stat(fail_sc,    key)
        delta = fp - cp
        print(f"  {label:<28} {ch}/{cn}={cp:.1%} [{clo:.1%},{chi:.1%}]"
              f"  {fh}/{fn}={fp:.1%} [{flo:.1%},{fhi:.1%}]  {delta:+.1%}")

    # ── Baselines ──
    print(f"\n{SEP}")
    print("BASELINES")
    print(SEP)
    uniform_single = 1 / N_CELLS
    uniform_either = 1 - (1 - uniform_single) ** 2
    print(f"  Uniform (single cell):  {uniform_single:.1%}")
    print(f"  Uniform (either cell):  {uniform_either:.1%}")

    # Always-center baseline
    always_center_subj_c = sum(1 for r in correct_sc if "center" in r["subj_cells"]) / max(len(correct_sc),1)
    always_center_subj_f = sum(1 for r in fail_sc    if "center" in r["subj_cells"]) / max(len(fail_sc),1)
    always_center_obj_c  = sum(1 for r in correct_sc if "center" in r["obj_cells"])  / max(len(correct_sc),1)
    always_center_obj_f  = sum(1 for r in fail_sc    if "center" in r["obj_cells"])  / max(len(fail_sc),1)
    always_center_either_c = sum(1 for r in correct_sc
                                 if "center" in r["subj_cells"] or "center" in r["obj_cells"]) / max(len(correct_sc),1)
    always_center_either_f = sum(1 for r in fail_sc
                                 if "center" in r["subj_cells"] or "center" in r["obj_cells"]) / max(len(fail_sc),1)
    print(f"\n  Always-tag-center accuracy (subject):  correct={always_center_subj_c:.1%}  fail={always_center_subj_f:.1%}")
    print(f"  Always-tag-center accuracy (object):   correct={always_center_obj_c:.1%}  fail={always_center_obj_f:.1%}")
    print(f"  Always-tag-center accuracy (either):   correct={always_center_either_c:.1%}  fail={always_center_either_f:.1%}")

    print(f"\n  Computing permuted-tag null ({args.n_perm} permutations)...")
    null = permuted_null(rows, args.n_perm)
    print(f"  Permuted null (subject):  correct={null['hit_subj']['correct']:.1%}  fail={null['hit_subj']['fail']:.1%}")
    print(f"  Permuted null (object):   correct={null['hit_obj']['correct']:.1%}  fail={null['hit_obj']['fail']:.1%}")
    print(f"  Permuted null (either):   correct={null['hit_either']['correct']:.1%}  fail={null['hit_either']['fail']:.1%}")

    # ── Center-specific breakdown ──
    print(f"\n{SEP}")
    print("CENTER CELL — IS THE CENTER BIAS VERIDICAL?")
    print(SEP)
    # items where subject/object actually IS in center
    subj_in_center_c = [r for r in correct_sc if "center" in r["subj_cells"]]
    subj_in_center_f = [r for r in fail_sc     if "center" in r["subj_cells"]]
    subj_not_center_c = [r for r in correct_sc if r["subj_cells"] and "center" not in r["subj_cells"]]
    subj_not_center_f = [r for r in fail_sc     if r["subj_cells"] and "center" not in r["subj_cells"]]
    print(f"\n  Subject IS in center cell:")
    print(f"    correct: {len(subj_in_center_c)} items, neutral=center: "
          f"{sum(1 for r in subj_in_center_c if r['neutral_tag']=='center')}/{len(subj_in_center_c)}")
    print(f"    fail:    {len(subj_in_center_f)} items, neutral=center: "
          f"{sum(1 for r in subj_in_center_f if r['neutral_tag']=='center')}/{len(subj_in_center_f)}")
    print(f"\n  Subject NOT in center cell:")
    print(f"    correct: {len(subj_not_center_c)} items, neutral=center: "
          f"{sum(1 for r in subj_not_center_c if r['neutral_tag']=='center')}/{len(subj_not_center_c)}"
          f" ({100*sum(1 for r in subj_not_center_c if r['neutral_tag']=='center')/max(len(subj_not_center_c),1):.1f}%)")
    print(f"    fail:    {len(subj_not_center_f)} items, neutral=center: "
          f"{sum(1 for r in subj_not_center_f if r['neutral_tag']=='center')}/{len(subj_not_center_f)}"
          f" ({100*sum(1 for r in subj_not_center_f if r['neutral_tag']=='center')/max(len(subj_not_center_f),1):.1f}%)")

    # ── Evidence cell distribution ──
    print(f"\n{SEP}")
    print("EVIDENCE CELL DISTRIBUTION (subject bbox)")
    print(SEP)
    all_subj_cells = [cell for r in rows for cell in r["subj_cells"]]
    dist = Counter(all_subj_cells)
    total_ev = sum(dist.values())
    for cell, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {cell:<20}: {cnt:4d} ({100*cnt/max(total_ev,1):.1f}%)")
    print(f"  [This is the 'true' spatial prior from COCO bbox statistics]")

    # ── Save enriched JSON ──
    out = args.results.replace(".json", "_coco.json")

    def jsonable(r):
        return {**r, "subj_cells": list(r["subj_cells"]), "obj_cells": list(r["obj_cells"])}

    with open(out, "w") as f:
        json.dump([jsonable(r) for r in rows], f, indent=2)
    print(f"\nEnriched results saved to {out}")


if __name__ == "__main__":
    main()
