"""
Analyze CF masking results: tag-vs-CF-GT accuracy with all required nulls.

Computes:
  - Tag-vs-CF-GT hit rate (neutral + stateless)
  - Permuted-tag null at causal cell
  - Always-center accuracy at causal cell
  - Stratified by center vs. non-center tagged (50/50 design)
  - Stratified by center vs. non-center CF-GT
  - Clean single-cell subset reported separately

Usage:
  python3 analyze_cf100.py experiments/real_vsr_v3_t0_cf100.json
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

random.seed(42)


def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / d
    return max(0, c-m), min(1, c+m)


def fmt(k, n):
    if n == 0: return "—"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n}={100*k/n:.1f}% [{100*lo:.1f}%,{100*hi:.1f}%]"


def permuted_null(items, tag_key, n_perm=2000):
    """Shuffle `tag_key` across items; return mean hit rate at cf_gt."""
    tags = [r[tag_key] for r in items if r.get(tag_key)]
    hits = []
    for _ in range(n_perm):
        shuffled = tags.copy(); random.shuffle(shuffled)
        tag_map = {r["image_id"]: t for r, t in zip(items, shuffled)}
        h = sum(1 for r in items if tag_map.get(r["image_id"]) == r["cf_gt"])
        hits.append(h / len(items) if items else 0)
    return sum(hits) / len(hits)


def always_center_rate(items):
    """Fraction of items where CF-GT == center."""
    return sum(1 for r in items if r["cf_gt"] == "center") / len(items) if items else 0


def report_block(label, items, tag_key):
    n = len(items)
    hits = sum(1 for r in items if r.get(tag_key) == r["cf_gt"])
    lo, hi = wilson_ci(hits, n)
    perm = permuted_null(items, tag_key)
    ac = always_center_rate(items)
    print(f"  {label}")
    print(f"    tag==CF-GT:      {fmt(hits, n)}")
    print(f"    permuted null:   {100*perm:.1f}%")
    print(f"    always-center:   {100*ac:.1f}%")
    print(f"    excess over null:{100*(hits/n - perm) if n else 0:+.1f}pp")


def main():
    os.chdir(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?",
                        default="experiments/real_vsr_v3_t0_cf100.json")
    args = parser.parse_args()

    # Try partial file if final not ready
    path = args.results
    if not os.path.exists(path):
        partial = path + ".partial"
        if os.path.exists(partial):
            print(f"(using partial: {partial})")
            path = partial
        else:
            print(f"File not found: {path}"); return

    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} CF results from {path}\n")

    # ── Outcome distribution ──
    outcomes = Counter(r["cf_gt"] for r in data)
    print("CF-GT outcome distribution:")
    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:3d}  ({100*v/len(data):.1f}%)")

    clean = [r for r in data if r["cf_gt"] not in ("none_found", "multiple", None)]
    print(f"\nClean single-cell CF-GT: {len(clean)}/{len(data)} ({100*len(clean)/len(data):.1f}%)")
    if not clean:
        print("No clean cases — nothing to score."); return

    SEP = "=" * 65
    # ── Main results: clean cases ──
    print(f"\n{SEP}")
    print("CLEAN SINGLE-CELL CF-GT CASES  (n={})".format(len(clean)))
    print(SEP)
    for tag_key, label in [("neutral_tagged_region", "neutral tag (Cond. A)"),
                             ("tagged_region",         "stateless tag (Cond. B)")]:
        print(f"\n  [{label}]")
        report_block("All clean", clean, tag_key)

        center_tagged  = [r for r in clean if r.get(tag_key) == "center"]
        noncenter_tagged = [r for r in clean if r.get(tag_key) != "center"]
        center_cfgt    = [r for r in clean if r["cf_gt"] == "center"]
        noncenter_cfgt = [r for r in clean if r["cf_gt"] != "center"]

        if center_cfgt or noncenter_cfgt:
            print()
            report_block(f"  CF-GT IS center (n={len(center_cfgt)})", center_cfgt, tag_key)
            report_block(f"  CF-GT NOT center (n={len(noncenter_cfgt)})", noncenter_cfgt, tag_key)

        if center_tagged or noncenter_tagged:
            print()
            report_block(f"  Tag IS center (n={len(center_tagged)})", center_tagged, tag_key)
            report_block(f"  Tag NOT center (n={len(noncenter_tagged)})", noncenter_tagged, tag_key)

    # ── Stratify by sampling stratum: center-tagged vs non-center-tagged subset ──
    print(f"\n{SEP}")
    print("STRATIFIED BY SAMPLING STRATUM (50/50 design)")
    print(SEP)
    # We sampled 50 center-stateless-tagged and 50 non-center-stateless-tagged
    stratum_center    = [r for r in clean if r.get("tagged_region") == "center"]
    stratum_noncenter = [r for r in clean if r.get("tagged_region") != "center"]
    print(f"\n  Stratum: stateless-tagged=center (n={len(stratum_center)} clean):")
    if stratum_center:
        hits = sum(1 for r in stratum_center if r["tagged_region"] == r["cf_gt"])
        print(f"    stateless tag==CF-GT: {fmt(hits, len(stratum_center))}")
        hits_n = sum(1 for r in stratum_center if r["neutral_tagged_region"] == r["cf_gt"])
        print(f"    neutral   tag==CF-GT: {fmt(hits_n, len(stratum_center))}")
        perm = permuted_null(stratum_center, "tagged_region")
        ac = always_center_rate(stratum_center)
        print(f"    permuted null: {100*perm:.1f}%  always-center: {100*ac:.1f}%")
        print(f"    CF-GT dist: {dict(Counter(r['cf_gt'] for r in stratum_center))}")

    print(f"\n  Stratum: stateless-tagged≠center (n={len(stratum_noncenter)} clean):")
    if stratum_noncenter:
        hits = sum(1 for r in stratum_noncenter if r["tagged_region"] == r["cf_gt"])
        print(f"    stateless tag==CF-GT: {fmt(hits, len(stratum_noncenter))}")
        hits_n = sum(1 for r in stratum_noncenter if r["neutral_tagged_region"] == r["cf_gt"])
        print(f"    neutral   tag==CF-GT: {fmt(hits_n, len(stratum_noncenter))}")
        perm = permuted_null(stratum_noncenter, "tagged_region")
        ac = always_center_rate(stratum_noncenter)
        print(f"    permuted null: {100*perm:.1f}%  always-center: {100*ac:.1f}%")
        print(f"    CF-GT dist: {dict(Counter(r['cf_gt'] for r in stratum_noncenter))}")

    # ── Summary table for paper ──
    print(f"\n{SEP}")
    print("PAPER TABLE (clean cases, both tags)")
    print(SEP)
    print(f"{'Metric':<45} {'Neutral (A)':>16} {'Stateless (B)':>16}")
    print("-" * 78)
    for subset_label, subset in [
        ("All clean", clean),
        ("  CF-GT=center",    [r for r in clean if r["cf_gt"]=="center"]),
        ("  CF-GT≠center",    [r for r in clean if r["cf_gt"]!="center"]),
    ]:
        for tag_key, col_label in [("neutral_tagged_region", "N"), ("tagged_region", "S")]:
            hits = sum(1 for r in subset if r.get(tag_key) == r["cf_gt"])
            n = len(subset)
            if col_label == "N":
                row_start = f"{subset_label:<45}"
                neutral_str = fmt(hits, n)
            else:
                stateless_str = fmt(hits, n)
        perm_n = permuted_null(subset, "neutral_tagged_region")
        perm_s = permuted_null(subset, "tagged_region")
        ac = always_center_rate(subset)
        print(f"{row_start} {neutral_str:>16} {stateless_str:>16}")
        print(f"  {'permuted null':43} {100*perm_n:>15.1f}% {100*perm_s:>15.1f}%")
        print(f"  {'always-center':43} {100*ac:>15.1f}% {100*ac:>15.1f}%")

    # ── Power / sufficiency check ──
    print(f"\n{SEP}")
    print("POWER CHECK")
    print(SEP)
    # Wilson CI half-width for the clean subset
    if clean:
        k = sum(1 for r in clean if r.get("tagged_region") == r["cf_gt"])
        n = len(clean)
        lo, hi = wilson_ci(k, n)
        ci_hw = (hi - lo) / 2
        print(f"  Clean n={n}, CI half-width ≈ ±{100*ci_hw:.0f}pp")
        if n < 20:
            remaining_ids = {r["image_id"] for r in data}
            print(f"\n  *** INSUFFICIENT CLEAN CASES (n={n} < 20) ***")
            print(f"  Recommendation: run a second CF batch on the remaining")
            print(f"  failures not yet masked (load real_vsr_v3_t0.json, exclude")
            print(f"  image_ids already in this file: {len(remaining_ids)} done).")
            print(f"  Target: ~50 additional masked failures to reach n≥20 clean.")
        else:
            print(f"  n={n} ≥ 20 — sufficient for ±{100*ci_hw:.0f}pp claim.")



if __name__ == "__main__":
    main()
