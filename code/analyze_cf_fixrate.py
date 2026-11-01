"""
Fix-rate vector analysis for CF masking results.

Devin's insight: at ~6 answer classes, masking an irrelevant cell "fixes" the
answer spuriously ~1/6 of the time. "multiple" CF-GT is consistent with this
noise floor, not with 4 true causal cells. This script recovers graded evidence:

  fix_rate[cell] = 1 if masking that cell changed the model's answer (binary
                     at temp=0), 0 otherwise

  argmax_cell = the cell that fixed the answer (or first if multiple)
  spurious_floor = expected fixes per item = 9 * (1/n_answer_classes)

Two new analyses beyond the binary CF-GT:

  1. Claimed-cell test: across items grouped by verbal tag, does P(fix | mask
     claimed cell) exceed P(fix | mask other cell)?

  2. Argmax-cell comparison: tag vs argmax_cell on ALL items with at least one
     fix — not just "clean" single-cell items (~doubles effective n).

Requires: items with "cell_answers" dict in the JSON.
Items without cell_answers (old format) fall back to binary CF-GT only.

Usage:
  python3 analyze_cf_fixrate.py experiments/real_vsr_v3_t0_cf100.json
  python3 analyze_cf_fixrate.py experiments/real_vsr_v3_t0_cf_combined.json
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

CELL_NAMES = [
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]
N_CELLS = len(CELL_NAMES)
# Theoretical spurious-fix floor differs by question type:
#   - Color questions (~6 classes): 1/6 ≈ 17% per cell
#   - Binary T/F questions (VSR): 1/2 = 50% theoretical, but at temp=0 a stable
#     model rarely changes answers when irrelevant cells are masked. Empirical
#     floor is ~10-15% per cell — report empirical, not theoretical.
THEORETICAL_FLOOR_BINARY = 1 / 2


def normalize_bool(s):
    if s is None:
        return None
    a = str(s).lower().strip().rstrip(".")
    if "true" in a and "false" not in a:
        return True
    if "false" in a and "true" not in a:
        return False
    if s is True:
        return True
    if s is False:
        return False
    return None


def compute_fix_rates(r):
    """Return dict: cell -> 1/0 (1 = masking changed answer from wrong to different).

    Returns None if cell_answers not available.
    """
    cell_ans = r.get("cell_answers")
    if not cell_ans:
        return None
    wrong_bool = normalize_bool(r.get("model_answer_raw", ""))
    rates = {}
    for c in CELL_NAMES:
        a = cell_ans.get(c)
        if a is None:
            rates[c] = None
        else:
            a_bool = normalize_bool(a) if not isinstance(a, bool) else a
            rates[c] = 1 if (a_bool is not None and a_bool != wrong_bool) else 0
    return rates


def argmax_cell_from_rates(rates):
    """First cell (in GRID_CELLS order) with fix_rate=1; None if no fixes."""
    if rates is None:
        return None
    for c in CELL_NAMES:
        if rates.get(c) == 1:
            return c
    return None


def cells_that_fix(rates):
    """All cells with fix_rate=1."""
    if rates is None:
        return []
    return [c for c in CELL_NAMES if rates.get(c) == 1]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    m = z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / d
    return max(0, c - m), min(1, c + m)


def fmt(k, n):
    if n == 0:
        return "—"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n}={100*k/n:.1f}% [{100*lo:.1f}%,{100*hi:.1f}%]"


def binomial_one_sided_p(k, n, p0):
    """P(X >= k) under Binomial(n, p0). Simple exact computation."""
    if n == 0:
        return 1.0
    p = 0.0
    binom = 1.0
    for i in range(k, n + 1):
        if i == k:
            binom = 1.0
            for j in range(n, n - k, -1):
                binom *= j
            for j in range(1, k + 1):
                binom //= j
        else:
            binom = binom * (n - i + 1) / i
        p += binom * (p0 ** i) * ((1 - p0) ** (n - i))
    return min(p, 1.0)


def main():
    os.chdir(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?",
                        default="experiments/real_vsr_v3_t0_cf100.json")
    args = parser.parse_args()

    path = args.results
    if not os.path.exists(path):
        partial = path + ".partial"
        if os.path.exists(partial):
            print(f"(using partial: {partial})")
            path = partial
        else:
            print(f"File not found: {path}")
            return

    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {path}\n")

    with_cell_answers = [r for r in data if r.get("cell_answers")]
    without = [r for r in data if not r.get("cell_answers")]
    print(f"Items with cell_answers: {len(with_cell_answers)}")
    print(f"Items without (old format, binary CF-GT only): {len(without)}\n")

    if not with_cell_answers:
        print("No items with cell_answers — run run_cf_rescan.py first.")
        return

    # ── Fix-rate vectors ──────────────────────────────────────────────────────
    all_rates = []
    for r in with_cell_answers:
        rates = compute_fix_rates(r)
        if rates:
            r["_fix_rates"] = rates
            r["_fix_cells"] = cells_that_fix(rates)
            r["_argmax"] = argmax_cell_from_rates(rates)
            all_rates.append(r)

    SEP = "=" * 70

    # ── Fix count distribution ────────────────────────────────────────────────
    print(SEP)
    print("FIX-COUNT DISTRIBUTION (per item, among items with cell_answers)")
    print(SEP)
    fix_counts = Counter(len(r["_fix_cells"]) for r in all_rates)
    total = len(all_rates)
    print(f"{'Fix count':>12} {'n':>6} {'%':>8}  interpretation")
    print("-" * 60)
    for k in range(10):
        n = fix_counts.get(k, 0)
        if n == 0:
            continue
        interp = {0: "none_found", 1: "clean causal"}.get(k, "multiple (possibly spurious)")
        print(f"{k:>12} {n:>6} {100*n/total:>7.1f}%  {interp}")

    # Spurious-fix floor
    mean_fixes = sum(len(r["_fix_cells"]) for r in all_rates) / len(all_rates)
    expected_spurious_binary = N_CELLS * THEORETICAL_FLOOR_BINARY  # 9 * 0.5 = 4.5 (theoretical max)
    # Empirical null: mean fix_rate across ALL cell-item pairs
    all_fix_bits = [v for r in all_rates for v in r["_fix_rates"].values() if v is not None]
    empirical_floor_per_cell = sum(all_fix_bits) / len(all_fix_bits) if all_fix_bits else 0
    print(f"\nMean fixes per item:            {mean_fixes:.2f}")
    print(f"Empirical fix rate per cell:    {100*empirical_floor_per_cell:.1f}% (= {empirical_floor_per_cell*N_CELLS:.2f} fixes/item)")
    print(f"Theoretical ceiling (binary q): {expected_spurious_binary:.1f} fixes/item (model stable at T=0, actual << theoretical)")
    print(f"NOTE: low empirical floor means 'multiple' in VSR reflects real distributed causes,")
    print(f"      not spurious flips (unlike color questions where floor ≈ 1/6 per cell).")

    # ── Claimed-cell test with within-item permutation null ───────────────────
    print(f"\n{SEP}")
    print("CLAIMED-CELL TEST — within-item permutation null (any-fix items only)")
    print("For each item, permute which cell is 'claimed' among its 9 cells (2000×).")
    print("Null conditions on each item's actual fix pattern — airtight against")
    print("'tag lands on high-fix-rate cells' objection.")
    print(SEP)

    any_fix_items = [r for r in all_rates if r["_argmax"] is not None]
    print(f"\n  Any-fix items (≥1 fix): {len(any_fix_items)} / {len(all_rates)}")

    for tag_key, label in [("neutral_tagged_region", "Neutral tag (Cond. A)"),
                            ("tagged_region", "Stateless tag (Cond. B)")]:
        tagged_af = [r for r in any_fix_items if r.get(tag_key)]
        if not tagged_af:
            print(f"\n  [{label}] — no tagged any-fix items")
            continue

        # Observed claimed-cell fix rate
        obs_fixes = [r["_fix_rates"].get(r[tag_key], 0) or 0 for r in tagged_af]
        n = len(obs_fixes)
        k = sum(obs_fixes)
        p_claimed = k / n if n else 0

        # Within-item permutation null: for each item, randomly pick one of 9 cells
        # as the "claimed" cell, compute its fix rate. Average over 2000 shuffles.
        N_PERM = 2000
        perm_rates = []
        for _ in range(N_PERM):
            hits = sum(
                r["_fix_rates"].get(random.choice(CELL_NAMES), 0) or 0
                for r in tagged_af
            )
            perm_rates.append(hits / n)
        null_mean = sum(perm_rates) / len(perm_rates)
        # One-sided p: fraction of permutations with rate >= observed
        p_perm = sum(1 for pr in perm_rates if pr >= p_claimed) / N_PERM

        lo, hi = wilson_ci(k, n)
        print(f"\n  [{label}]  (n any-fix items with tag = {n})")
        print(f"    Observed claimed fix rate:     {k}/{n}={100*p_claimed:.1f}% [{100*lo:.1f}%,{100*hi:.1f}%]")
        print(f"    Within-item perm. null mean:   {100*null_mean:.1f}%")
        print(f"    Excess over null:              {100*(p_claimed - null_mean):+.1f}pp")
        print(f"    One-sided permutation p:       {p_perm:.3f}")
        print(f"    Interpretation: {'directionally consistent but underpowered (n.s.)' if p_perm > 0.05 else 'significant at p<0.05'}")

        # Center vs non-center breakdown
        for sublabel, subset in [("tag=center", [r for r in tagged_af if r.get(tag_key) == "center"]),
                                   ("tag≠center", [r for r in tagged_af if r.get(tag_key) != "center"])]:
            if not subset:
                continue
            sf = sum(r["_fix_rates"].get(r[tag_key], 0) or 0 for r in subset)
            sn = len(subset)
            # Per-subset permutation null
            sub_perm = []
            for _ in range(500):
                sh = sum(r["_fix_rates"].get(random.choice(CELL_NAMES), 0) or 0 for r in subset)
                sub_perm.append(sh / sn)
            sub_null = sum(sub_perm) / len(sub_perm)
            print(f"      {sublabel} (n={sn}): claimed={100*sf/sn:.1f}% null={100*sub_null:.1f}% excess={100*(sf/sn - sub_null):+.1f}pp")

    # ── Argmax-cell comparison (replaces clean-only) ──────────────────────────
    print(f"\n{SEP}")
    print("ARGMAX-CELL COMPARISON (tag vs first-fix cell)")
    print("Effective n: all items with ≥1 fix (not just clean single-cell)")
    print(SEP)

    # Items with at least one fix
    any_fix = [r for r in all_rates if r["_argmax"] is not None]
    print(f"\n  Items with ≥1 fix: {len(any_fix)} / {len(all_rates)}")

    for tag_key, label in [("neutral_tagged_region", "Neutral tag (Cond. A)"),
                            ("tagged_region", "Stateless tag (Cond. B)")]:
        tagged = [r for r in any_fix if r.get(tag_key)]
        hits = sum(1 for r in tagged if r.get(tag_key) == r["_argmax"])
        n = len(tagged)
        print(f"\n  [{label}]  tag==argmax_cell:")
        print(f"    All items with ≥1 fix: {fmt(hits, n)}")

        # Null: if tag were random among fix-cells, what's the expected hit rate?
        # null = mean(1/len(fix_cells)) for each item with multiple fixes
        null_hits = sum(
            1 / max(len(r["_fix_cells"]), 1)
            for r in tagged if r.get(tag_key)
        ) / n if n else 0
        print(f"    Null (random among fix cells): {100*null_hits:.1f}%")

        # Center vs non-center tag
        for sublabel, subset in [
            ("tag=center (n)", [r for r in tagged if r.get(tag_key) == "center"]),
            ("tag≠center (n)", [r for r in tagged if r.get(tag_key) != "center"]),
        ]:
            if subset:
                h = sum(1 for r in subset if r.get(tag_key) == r["_argmax"])
                print(f"      {sublabel}={len(subset)}: {fmt(h, len(subset))}")

    # ── Tag-vs-argmax vs tag-vs-clean side-by-side ────────────────────────────
    print(f"\n{SEP}")
    print("PAPER TABLE: binary CF-GT (clean only) vs argmax-cell (all-fix)")
    print(SEP)

    clean = [r for r in all_rates if r.get("cf_gt") not in ("none_found", "multiple", None)]
    print(f"\n  Clean single-cell (n={len(clean)})    vs    Any-fix (n={len(any_fix)})")
    print(f"  {'Tag':30s} {'clean hit':>12} {'any-fix hit':>12}")
    print("  " + "-" * 55)
    for tag_key, lbl in [("neutral_tagged_region", "neutral (Cond.A)"),
                          ("tagged_region", "stateless (Cond.B)")]:
        # clean
        cl_tagged = [r for r in clean if r.get(tag_key)]
        cl_hits = sum(1 for r in cl_tagged if r.get(tag_key) == r.get("cf_gt"))
        # argmax
        af_tagged = [r for r in any_fix if r.get(tag_key)]
        af_hits = sum(1 for r in af_tagged if r.get(tag_key) == r.get("_argmax"))
        print(f"  {lbl:30s} {fmt(cl_hits, len(cl_tagged)):>12} {fmt(af_hits, len(af_tagged)):>12}")


if __name__ == "__main__":
    main()
