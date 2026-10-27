"""
Synthetic test data generator for the Gap H pilot.

Creates simple controlled images where:
- Objects are placed in specific grid regions
- Questions have unambiguous correct answers
- The causal region is known by construction (no annotation needed)

This is better than VQAv2 for a pilot because:
- Zero download / zero cost
- Perfect ground truth about which region matters
- Controllable difficulty (we can make questions VLMs typically fail)

Run: python3 generate_synthetic.py
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

DATA_DIR = Path("./data")
IMAGES_DIR = DATA_DIR / "images"

# Colors with names
COLORS = {
    "red":    (220, 50,  50),
    "blue":   (50,  80,  220),
    "green":  (50,  180, 80),
    "yellow": (230, 200, 50),
    "purple": (150, 50,  200),
    "orange": (230, 130, 50),
}

SHAPES = ["circle", "square", "triangle"]

# Grid positions (3x3 grid on 300x300 image)
GRID_POSITIONS = {
    "top-left":      (50,  50),
    "top-center":    (150, 50),
    "top-right":     (250, 50),
    "middle-left":   (50,  150),
    "center":        (150, 150),
    "middle-right":  (250, 150),
    "bottom-left":   (50,  250),
    "bottom-center": (150, 250),
    "bottom-right":  (250, 250),
}

SIZE = 300
RADIUS = 30


def draw_shape(draw, shape, center, color, size=RADIUS):
    x, y = center
    if shape == "circle":
        draw.ellipse([x-size, y-size, x+size, y+size], fill=color, outline=(0,0,0), width=2)
    elif shape == "square":
        draw.rectangle([x-size, y-size, x+size, y+size], fill=color, outline=(0,0,0), width=2)
    elif shape == "triangle":
        draw.polygon([(x, y-size), (x-size, y+size), (x+size, y+size)], fill=color, outline=(0,0,0), width=2)


POS_LABELS = {
    "top-left":      "top-left",
    "top-center":    "top-center",
    "top-right":     "top-right",
    "middle-left":   "middle-left",
    "center":        "center",
    "middle-right":  "middle-right",
    "bottom-left":   "bottom-left",
    "bottom-center": "bottom-center",
    "bottom-right":  "bottom-right",
}

def make_relative_position(sample_id: int) -> dict:
    """
    Two-step spatial reasoning:
    1. Find a named reference object (UNIQUE color+shape, so unambiguous)
    2. Report the color of the object in an adjacent cell

    VLMs fail at step 1→2 when there are multiple candidates.
    The causal region is the TARGET cell (adjacent to reference).
    Reference object is always unique to prevent question ambiguity.
    """
    img = Image.new("RGB", (SIZE, SIZE), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    for i in [100, 200]:
        draw.line([(i, 0), (i, SIZE)], fill=(180, 180, 180), width=2)
        draw.line([(0, i), (SIZE, i)], fill=(180, 180, 180), width=2)

    adjacency = {
        "top-left":      {"below": "middle-left",  "right": "top-center"},
        "top-center":    {"below": "center",        "left":  "top-left",    "right": "top-right"},
        "top-right":     {"below": "middle-right",  "left":  "top-center"},
        "middle-left":   {"above": "top-left",      "below": "bottom-left", "right": "center"},
        "center":        {"above": "top-center",    "below": "bottom-center","left": "middle-left","right":"middle-right"},
        "middle-right":  {"above": "top-right",     "below": "bottom-right","left": "center"},
        "bottom-left":   {"above": "middle-left",   "right": "bottom-center"},
        "bottom-center": {"above": "center",        "left":  "bottom-left", "right": "bottom-right"},
        "bottom-right":  {"above": "middle-right",  "left":  "bottom-center"},
    }

    color_names = list(COLORS.keys())
    shape_names = list(SHAPES)

    # Build grid with UNIQUE (color, shape) pairs — no duplicates
    combos = [(c, s) for c in color_names for s in shape_names]
    random.shuffle(combos)
    positions = list(GRID_POSITIONS.items())
    random.shuffle(positions)

    grid = {}
    for (pos_name, coords), (color, shape) in zip(positions, combos[:9]):
        draw_shape(draw, shape, coords, COLORS[color])
        grid[pos_name] = (color, shape)

    # Pick a reference cell that has at least one neighbour
    ref_pos = random.choice([p for p in adjacency if adjacency[p]])
    direction = random.choice(list(adjacency[ref_pos].keys()))
    target_pos = adjacency[ref_pos][direction]

    ref_color, ref_shape   = grid[ref_pos]
    target_color, target_shape = grid[target_pos]

    question      = f"What color is the {target_shape} that is directly {direction} of the {ref_color} {ref_shape}?"
    answer        = target_color
    causal_region = target_pos

    image_file = f"synthetic_{sample_id:04d}.png"
    img.save(str(IMAGES_DIR / image_file))

    return {
        "image_id":       f"synthetic_{sample_id:04d}",
        "image_file":     image_file,
        "question":       question,
        "answer":         answer,
        "question_type":  "relative_position",
        "causal_region":  causal_region,
        "ref_pos":        ref_pos,
        "direction":      direction,
        "grid_layout":    {pos: {"color": c, "shape": s} for pos, (c, s) in grid.items()},
    }


def make_adjacent_distractor(sample_id: int) -> dict:
    """
    Single-distractor design for clean counterfactual GTs.

    Layout:
    - 9-cell grid, all cells filled with unique (color, shape) combos
    - TARGET cell: has the shape being asked about, with the CORRECT color
    - ADJACENT DISTRACTOR: a neighboring cell with the SAME SHAPE but a DIFFERENT color
    - Question: "What color is the {shape} in the {target_pos}?"
    - Common failure: model answers with the distractor's color

    Counterfactual GT: masking the distractor cell should fix the error
    (the model stops being confused by the competing same-shape nearby)

    Known ground truth:
    - causal_region: the target cell (correct answer location)
    - distractor_region: the adjacent same-shape cell (what causes confusion)
    """
    img = Image.new("RGB", (SIZE, SIZE), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    for i in [100, 200]:
        draw.line([(i, 0), (i, SIZE)], fill=(180, 180, 180), width=2)
        draw.line([(0, i), (SIZE, i)], fill=(180, 180, 180), width=2)

    # Adjacency: cell → list of directly adjacent cells (share an edge)
    adjacency = {
        "top-left":     ["top-center", "middle-left"],
        "top-center":   ["top-left", "top-right", "center"],
        "top-right":    ["top-center", "middle-right"],
        "middle-left":  ["top-left", "center", "bottom-left"],
        "center":       ["top-center", "middle-left", "middle-right", "bottom-center"],
        "middle-right": ["top-right", "center", "bottom-right"],
        "bottom-left":  ["middle-left", "bottom-center"],
        "bottom-center":["bottom-left", "bottom-right", "center"],
        "bottom-right": ["middle-right", "bottom-center"],
    }

    # Build grid with unique (color, shape) pairs
    combos = [(c, s) for c in COLORS for s in SHAPES]
    random.shuffle(combos)
    positions = list(GRID_POSITIONS.items())
    random.shuffle(positions)

    grid = {}
    for (pos_name, coords), (color, shape) in zip(positions, combos[:9]):
        grid[pos_name] = (color, shape, coords)

    # Pick a target cell and inject an adjacent same-shape distractor
    target_pos = random.choice(list(grid.keys()))
    target_color, target_shape, target_coords = grid[target_pos]

    neighbors = adjacency[target_pos]
    distractor_pos = random.choice(neighbors)

    # Override distractor with same shape, different color
    distractor_color = random.choice([c for c in COLORS if c != target_color])
    _, _, distractor_coords = grid[distractor_pos]
    grid[distractor_pos] = (distractor_color, target_shape, distractor_coords)

    # Draw all cells
    for pos_name, (color, shape, coords) in grid.items():
        draw_shape(draw, shape, coords, COLORS[color])

    question      = f"What color is the {target_shape} in the {target_pos} of the image?"
    answer        = target_color
    causal_region = target_pos

    image_file = f"synthetic_{sample_id:04d}.png"
    img.save(str(IMAGES_DIR / image_file))

    return {
        "image_id":         f"synthetic_{sample_id:04d}",
        "image_file":       image_file,
        "question":         question,
        "answer":           answer,
        "question_type":    "adjacent_distractor",
        "causal_region":    causal_region,
        "distractor_region": distractor_pos,
        "distractor_color": distractor_color,
        "grid_layout":      {pos: {"color": c, "shape": s}
                             for pos, (c, s, _) in grid.items()},
    }



    """
    9-cell grid, all cells filled.
    Hard variant: one COLUMN has the SAME SHAPE in all 3 rows (different colors).
    Question: 'What color is the {shape} in the {target_pos}?'
    Same-shape neighbors force precise row-level localization — VLMs' key failure point.
    """
    img = Image.new("RGB", (SIZE, SIZE), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    for i in [100, 200]:
        draw.line([(i, 0), (i, SIZE)], fill=(180, 180, 180), width=2)
        draw.line([(0, i), (SIZE, i)], fill=(180, 180, 180), width=2)

    # Column structure: left / center / right
    columns = {
        "left":   [("top-left",      (50,  50)),  ("middle-left",  (50,  150)), ("bottom-left",  (50,  250))],
        "center": [("top-center",    (150, 50)),  ("center",       (150, 150)), ("bottom-center",(150, 250))],
        "right":  [("top-right",     (250, 50)),  ("middle-right", (250, 150)), ("bottom-right", (250, 250))],
    }

    repeated_col = random.choice(["left", "center", "right"])
    repeated_shape = random.choice(SHAPES)
    col_colors = random.sample(list(COLORS.keys()), 3)  # 3 distinct colors for the repeated column

    grid = {}

    # Fill the repeated column with same shape, different colors
    for (pos_name, coords), color in zip(columns[repeated_col], col_colors):
        draw_shape(draw, repeated_shape, coords, COLORS[color])
        grid[pos_name] = (color, repeated_shape)

    # Fill other columns with different shapes
    other_colors = list(COLORS.keys())
    other_shapes = [s for s in SHAPES if s != repeated_shape]
    other_idx = 0
    for col_key, col_cells in columns.items():
        if col_key == repeated_col:
            continue
        for pos_name, coords in col_cells:
            color = other_colors[other_idx % len(other_colors)]
            shape = other_shapes[other_idx % len(other_shapes)]
            draw_shape(draw, shape, coords, COLORS[color])
            grid[pos_name] = (color, shape)
            other_idx += 1

    # Target: the MIDDLE row of the repeated column (hardest to localize)
    row_cells = {
        "left":   [("top-left", "middle-left", "bottom-left")],
        "center": [("top-center", "center", "bottom-center")],
        "right":  [("top-right", "middle-right", "bottom-right")],
    }
    _, target_pos, _ = row_cells[repeated_col][0]
    target_color, target_shape = grid[target_pos]

    question      = f"What color is the {target_shape} in the {target_pos} of the image?"
    answer        = target_color
    causal_region = target_pos

    image_file = f"synthetic_{sample_id:04d}.png"
    img.save(str(IMAGES_DIR / image_file))

    return {
        "image_id":      f"synthetic_{sample_id:04d}",
        "image_file":    image_file,
        "question":      question,
        "answer":        answer,
        "question_type": "dense_grid",
        "causal_region": causal_region,
        "repeated_col":  repeated_col,
        "grid_layout":   {pos: {"color": c, "shape": s} for pos, (c, s) in grid.items()},
    }


def make_sample(sample_id: int, question_type: str) -> dict:
    """Generate one synthetic image + QA pair with known causal region."""
    if question_type == "adjacent_distractor":
        return make_adjacent_distractor(sample_id)
    if question_type == "relative_position":
        return make_relative_position(sample_id)

    img = Image.new("RGB", (SIZE, SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw a subtle grid for reference
    for i in [100, 200]:
        draw.line([(i, 0), (i, SIZE)], fill=(200, 200, 200), width=1)
        draw.line([(0, i), (SIZE, i)], fill=(200, 200, 200), width=1)

    if question_type == "color_left_right":
        # Two objects, ask about left one
        left_color  = random.choice(list(COLORS.keys()))
        right_color = random.choice([c for c in COLORS if c != left_color])
        left_shape  = random.choice(SHAPES)
        right_shape = random.choice(SHAPES)

        draw_shape(draw, left_shape,  (75, 150),  COLORS[left_color])
        draw_shape(draw, right_shape, (225, 150), COLORS[right_color])

        question       = f"What color is the {left_shape} on the left side of the image?"
        answer         = left_color
        causal_region  = "middle-left"
        distractor_region = "middle-right"

    elif question_type == "count":
        # N objects, ask how many
        n = random.choice([2, 3, 4])
        positions = random.sample(list(GRID_POSITIONS.values()), n)
        color = random.choice(list(COLORS.keys()))
        shape = random.choice(SHAPES)
        for pos in positions:
            draw_shape(draw, shape, pos, COLORS[color])

        question      = f"How many {color} {shape}s are in the image?"
        answer        = str(n)
        causal_region = "all"
        distractor_region = None

    elif question_type == "top_bottom":
        # Object on top and bottom, ask about top
        top_color    = random.choice(list(COLORS.keys()))
        bottom_color = random.choice([c for c in COLORS if c != top_color])
        shape        = random.choice(SHAPES)

        draw_shape(draw, shape, (150, 60),  COLORS[top_color])
        draw_shape(draw, shape, (150, 240), COLORS[bottom_color])

        question       = f"What color is the {shape} at the top of the image?"
        answer         = top_color
        causal_region  = "top-center"
        distractor_region = "bottom-center"

    elif question_type == "spatial_relation":
        pos_a  = random.choice(["top-left", "middle-left", "bottom-left"])
        pos_b  = random.choice(["top-right", "middle-right", "bottom-right"])
        color_a = random.choice(list(COLORS.keys()))
        color_b = random.choice([c for c in COLORS if c != color_a])

        draw_shape(draw, "circle", GRID_POSITIONS[pos_a], COLORS[color_a])
        draw_shape(draw, "square", GRID_POSITIONS[pos_b], COLORS[color_b])

        question      = f"Is the {color_a} circle to the left of the {color_b} square?"
        answer        = "yes"
        causal_region = pos_a

    else:
        raise ValueError(f"Unknown question_type: {question_type}")

    image_file = f"synthetic_{sample_id:04d}.png"
    img.save(str(IMAGES_DIR / image_file))

    return {
        "image_id":           f"synthetic_{sample_id:04d}",
        "image_file":         image_file,
        "question":           question,
        "answer":             answer,
        "question_type":      question_type,
        "causal_region":      causal_region,
        "distractor_region":  locals().get("distractor_region"),
    }


def generate(n_samples: int = 200, output_path: str = "data/qa_samples.json"):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # adjacent_distractor: cleanest design for Gap H — single-cell counterfactual GT
    #   each image has one same-shape distractor adjacent to the target cell
    #   when model fails, masking the distractor should uniquely fix the error
    # relative_position: two-step reasoning — higher error rate, more complex errors
    # dense_grid: same-shape column — moderate error rate
    types = [
        "adjacent_distractor", "adjacent_distractor", "adjacent_distractor",
        "adjacent_distractor", "relative_position", "relative_position",
        "color_left_right", "top_bottom", "relative_position",
    ]
    samples = []

    for i in range(n_samples):
        q_type = types[i % len(types)]
        try:
            sample = make_sample(i, q_type)
            samples.append(sample)
        except Exception as e:
            print(f"  Skipped sample {i}: {e}")

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Generated {len(samples)} synthetic samples → {output_path}")
    print(f"Images saved to {IMAGES_DIR}/")
    print(f"\nQuestion type breakdown:")
    from collections import Counter
    for t, c in Counter(s["question_type"] for s in samples).items():
        print(f"  {t}: {c}")
    print(f"\nNext: python3 pilot_experiment.py --n_samples 50")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output", default="data/qa_samples.json")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    generate(args.n_samples, args.output)
