"""Render the canonical Mermaid architecture source to directly viewable images.

The repository does not require a Mermaid CLI or browser. This renderer reads
the deliberately small flowchart subset used by MODEL_ARCHITECTURE_DIAGRAM.mmd
and produces deterministic SVG and PNG artifacts with a publication-friendly
layout. Labels and edges have exactly one source: the .mmd file.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "MODEL_ARCHITECTURE_DIAGRAM.mmd"
SVG_OUTPUT = ROOT / "MODEL_ARCHITECTURE_DIAGRAM.svg"
PNG_OUTPUT = ROOT / "MODEL_ARCHITECTURE_DIAGRAM.png"

NODE_PATTERN = re.compile(r'([A-Za-z][A-Za-z0-9_]*)\["([^"]*)"\]')
EDGE_PATTERN = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)(?:\[\"[^\"]*\"\])?\s*-->\s*"
    r"([A-Za-z][A-Za-z0-9_]*)(?:\[\"[^\"]*\"\])?\s*$"
)

# Stable, human-tuned layout. Node labels and graph edges remain sourced only
# from the Mermaid file. A newly added node fails closed until its placement is
# chosen deliberately instead of silently producing an unreadable diagram.
POSITIONS: dict[str, tuple[float, float]] = {
    "A": (-5.2, 12.4),
    "AF": (-1.5, 12.4),
    "T": (3.0, 12.4),
    "FP": (6.7, 12.4),
    "E": (-5.2, 10.2),
    "FM": (-1.5, 10.2),
    "C": (3.0, 10.2),
    "EX": (6.7, 10.2),
    "JM": (-3.35, 8.0),
    "NE": (3.0, 8.0),
    "IE": (-3.35, 5.8),
    "E2": (3.0, 5.8),
    "SC": (4.85, 3.6),
    "CE": (4.85, 1.4),
    "CAT": (0.0, -0.5),
    "TR": (0.0, -2.7),
    "FN": (0.0, -4.7),
    "OH": (0.0, -6.7),
    "FULL": (0.0, -8.7),
    "SLICE": (0.0, -10.7),
    "LOSS": (0.0, -12.7),
    "DEC": (0.0, -14.9),
    # Inference branch, hung off the canvas-logits slice to the right of the
    # training column. EBA sits beside SAMP because it feeds the next pass back
    # into it; the FIX -> TERM -> RET spine is the exit path.
    "SAMP": (6.9, -8.7),
    "FIX": (6.9, -10.9),
    "EBA": (11.1, -10.9),
    "TERM": (6.9, -13.1),
    "RET": (6.9, -15.3),
}

COLORS: dict[str, tuple[str, str]] = {
    "input": ("#E8F1FF", "#2563A6"),
    "feature": ("#F3E8FF", "#7C3AED"),
    "canvas": ("#FFF1DF", "#C76B00"),
    "conditioning": ("#E4F7F5", "#147D75"),
    "joint": ("#E8EEF8", "#334E7D"),
    "output": ("#E7F7E9", "#27783B"),
    "loss": ("#FCE8E8", "#B83232"),
}

CATEGORY = {
    "A": "input",
    "E": "input",
    "AF": "feature",
    "FM": "feature",
    "JM": "feature",
    "IE": "input",
    "T": "canvas",
    "C": "canvas",
    "NE": "canvas",
    "E2": "canvas",
    "FP": "conditioning",
    "EX": "conditioning",
    "SC": "conditioning",
    "CE": "conditioning",
    "CAT": "joint",
    "TR": "joint",
    "FN": "joint",
    "OH": "output",
    "FULL": "output",
    "SLICE": "output",
    "LOSS": "loss",
    "DEC": "loss",
    "SAMP": "canvas",
    "FIX": "conditioning",
    "EBA": "canvas",
    "TERM": "output",
    "RET": "output",
}


def parse_mermaid(source: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Parse node labels and simple directed edges from the canonical source."""

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("flowchart") or line.startswith("%%"):
            continue
        for node_id, raw_label in NODE_PATTERN.findall(line):
            nodes[node_id] = html.unescape(raw_label).replace("<br/>", "\n")
        match = EDGE_PATTERN.match(line)
        if match is None:
            raise ValueError(f"unsupported Mermaid line: {raw_line!r}")
        edges.append((match.group(1), match.group(2)))

    referenced = {node for edge in edges for node in edge}
    missing_labels = referenced - nodes.keys()
    if missing_labels:
        raise ValueError(f"nodes referenced before any label definition: {sorted(missing_labels)}")
    missing_positions = nodes.keys() - POSITIONS.keys()
    stale_positions = POSITIONS.keys() - nodes.keys()
    if missing_positions or stale_positions:
        raise ValueError(
            "diagram layout and Mermaid nodes differ: "
            f"missing_positions={sorted(missing_positions)}, "
            f"stale_positions={sorted(stale_positions)}"
        )
    return nodes, edges


def box_size(label: str) -> tuple[float, float]:
    """Return a readable box size from the label's line count and width."""

    lines = label.splitlines()
    width = min(4.0, max(2.7, max(len(line) for line in lines) * 0.095 + 0.8))
    height = 0.62 + 0.34 * len(lines)
    return width, height


def boundary_point(
    center: tuple[float, float],
    size: tuple[float, float],
    toward: tuple[float, float],
) -> tuple[float, float]:
    """Intersect a center-to-center line with a rectangular box boundary."""

    x, y = center
    tx, ty = toward
    dx, dy = tx - x, ty - y
    if dx == 0 and dy == 0:
        return center
    half_width, half_height = size[0] / 2, size[1] / 2
    scale_x = half_width / abs(dx) if dx else float("inf")
    scale_y = half_height / abs(dy) if dy else float("inf")
    scale = min(scale_x, scale_y)
    return x + dx * scale, y + dy * scale


def render() -> None:
    """Render the canonical Mermaid graph to SVG and PNG."""

    nodes, edges = parse_mermaid(SOURCE.read_text(encoding="utf-8"))
    sizes = {node_id: box_size(label) for node_id, label in nodes.items()}

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "thesis-ml-model-architecture",
        }
    )
    fig, axis = plt.subplots(figsize=(20.8, 23.6), dpi=150)
    axis.set_xlim(-7.25, 13.3)
    # Lower bound clears the decomposition node at y=-14.9 plus its box height
    # and leaves room for the source footer beneath it.
    axis.set_ylim(-16.7, 15.4)
    axis.axis("off")
    fig.patch.set_facecolor("#F8FAFC")
    axis.set_facecolor("#F8FAFC")

    axis.text(
        0.65,
        14.9,
        "Thesis_ML Model Architecture",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#172033",
    )
    axis.text(
        0.65,
        14.3,
        "Current smallTrainingTestV3 pipeline · 29,318,720 trainable parameters",
        ha="center",
        va="center",
        fontsize=12,
        color="#4A5568",
    )
    axis.text(-3.4, 13.6, "CLAMPED INPUT", ha="center", fontsize=10, fontweight="bold", color="#334E7D")
    axis.text(4.85, 13.6, "DENOISING CANVAS", ha="center", fontsize=10, fontweight="bold", color="#8A4B08")
    axis.text(8.0, -7.1, "ITERATIVE SAMPLING (INFERENCE)", ha="center", fontsize=10, fontweight="bold", color="#8A4B08")

    for source, target in edges:
        start = boundary_point(POSITIONS[source], sizes[source], POSITIONS[target])
        end = boundary_point(POSITIONS[target], sizes[target], POSITIONS[source])
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.7,
            color="#667085",
            connectionstyle="arc3,rad=0.0",
            zorder=1,
        )
        axis.add_patch(arrow)

    for node_id, label in nodes.items():
        x, y = POSITIONS[node_id]
        width, height = sizes[node_id]
        fill, border = COLORS[CATEGORY[node_id]]
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=2.0,
            edgecolor=border,
            facecolor=fill,
            zorder=2,
        )
        axis.add_patch(box)
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#172033",
            linespacing=1.25,
            zorder=3,
        )

    axis.text(
        0.65,
        -16.2,
        "Canonical graph source: MODEL_ARCHITECTURE_DIAGRAM.mmd",
        ha="center",
        va="center",
        fontsize=9,
        color="#667085",
    )

    fig.savefig(
        SVG_OUTPUT,
        format="svg",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        metadata={"Creator": "Thesis_ML Model_Architecture/render_diagram.py", "Date": None},
    )
    svg_text = SVG_OUTPUT.read_text(encoding="utf-8")
    SVG_OUTPUT.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        PNG_OUTPUT,
        format="png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        metadata={"Software": "Thesis_ML Model_Architecture/render_diagram.py"},
    )
    plt.close(fig)
    print(f"rendered {SVG_OUTPUT.name} and {PNG_OUTPUT.name} from {SOURCE.name}")


if __name__ == "__main__":
    render()
