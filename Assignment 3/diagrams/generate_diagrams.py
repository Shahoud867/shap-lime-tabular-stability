"""Generate editable .drawio XML sources and publication-grade PDF/400-DPI PNG exports
for the three structural/conceptual diagrams (Figs. 1, 2, 8) of Assignment 3.

Every coordinate in the .drawio XML and the vector matplotlib PDF/PNG renderer
is derived from the same layout specification so both remain synchronized.
All coordinates are in 1/100ths of an inch at IEEE double-column width (7.16 in),
so matplotlib fontsize=8.0..8.5 corresponds to exact >= 8.0 pt printed font size.
"""

from __future__ import annotations

import html
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

FONT_FAMILY = "DejaVu Sans"

PALETTE = {
    "base_fill": "#E8F3FA",
    "base_edge": "#0072B2",
    "base_text": "#0C3B5E",
    "ext_fill": "#E6F5F0",
    "ext_edge": "#009E73",
    "ext_text": "#0B4F3C",
    "warn_fill": "#FDF0E6",
    "warn_edge": "#D55E00",
    "warn_text": "#6E2E00",
    "amber_fill": "#FEF6E6",
    "amber_edge": "#E69F00",
    "amber_text": "#6B4700",
    "ctrl_fill": "#F1F5F9",
    "ctrl_edge": "#64748B",
    "ctrl_text": "#1E293B",
    "frame_base": "#F7FBFE",
    "frame_ext": "#F6FBF9",
}


@dataclass
class Box:
    id: str
    x: float
    y: float  # top-down coordinates (100 units = 1 inch)
    w: float
    h: float
    title: str
    subtitle: str = ""
    fill: str = "#FFFFFF"
    edge: str = "#333333"
    text_color: str = "#111111"
    dashed: bool = False
    rounded: bool = True
    is_container: bool = False
    is_diamond: bool = False
    lw: float = 1.25
    title_size: float = 8.3
    sub_size: float = 8.0


@dataclass
class Arrow:
    src_xy: tuple[float, float]
    dst_xy: tuple[float, float]
    src_id: str = ""
    dst_id: str = ""
    label: str = ""
    label_xy: Optional[tuple[float, float]] = None
    color: str = "#334155"
    dashed: bool = False
    lw: float = 1.2
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    label_size: float = 8.0
    label_bg: str = "#FFFFFF"


def _escape_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def write_drawio(
    out_path: Path,
    diagram_name: str,
    width: float,
    height: float,
    boxes: list[Box],
    arrows: list[Arrow],
) -> None:
    """Emit a valid diagrams.net (.drawio) mxGraph XML file."""
    mxfile = ET.Element("mxfile", host="app.diagrams.net", type="device")
    diagram = ET.SubElement(mxfile, "diagram", id=diagram_name, name=diagram_name)
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1000",
        dy="600",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth=str(int(round(width))),
        pageHeight=str(int(round(height))),
        background="#FFFFFF",
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    for b in boxes:
        if b.is_container:
            label = f"<b>{_escape_html(b.title)}</b>"
            style = (
                f"rounded=1;whiteSpace=wrap;html=1;verticalAlign=top;"
                f"align=left;spacingLeft=10;spacingTop=6;fontFamily=Helvetica;"
                f"fontSize={int(round(b.title_size))};fillColor={b.fill};"
                f"strokeColor={b.edge};fontColor={b.text_color};"
                f"strokeWidth={b.lw};dashed={'1' if b.dashed else '0'};"
            )
        elif b.is_diamond:
            label = f"<b>{_escape_html(b.title)}</b>"
            if b.subtitle:
                label += f"<br><font style='font-size:{int(round(b.sub_size))}px'>{_escape_html(b.subtitle)}</font>"
            style = (
                f"rhombus;whiteSpace=wrap;html=1;fontFamily=Helvetica;"
                f"fontSize={int(round(b.title_size))};fillColor={b.fill};"
                f"strokeColor={b.edge};fontColor={b.text_color};strokeWidth={b.lw};"
            )
        else:
            label = f"<b>{_escape_html(b.title)}</b>"
            if b.subtitle:
                label += f"<br><font style='font-size:{int(round(b.sub_size))}px'>{_escape_html(b.subtitle)}</font>"
            style = (
                f"rounded={'1' if b.rounded else '0'};whiteSpace=wrap;html=1;"
                f"fontFamily=Helvetica;fontSize={int(round(b.title_size))};"
                f"fillColor={b.fill};strokeColor={b.edge};fontColor={b.text_color};"
                f"strokeWidth={b.lw};dashed={'1' if b.dashed else '0'};"
            )
        cell = ET.SubElement(
            root,
            "mxCell",
            id=b.id,
            value=label,
            style=style,
            vertex="1",
            parent="1",
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            x=str(round(b.x, 1)),
            y=str(round(b.y, 1)),
            width=str(round(b.w, 1)),
            height=str(round(b.h, 1)),
            **{"as": "geometry"},
        )

    for idx, a in enumerate(arrows, start=1):
        style = (
            f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
            f"html=1;endArrow=classic;endFill=1;strokeColor={a.color};"
            f"strokeWidth={a.lw};dashed={'1' if a.dashed else '0'};"
            f"fontFamily=Helvetica;fontSize={int(round(a.label_size))};"
            f"fontColor={a.color};labelBackgroundColor={a.label_bg};"
        )
        attrs = {
            "id": f"edge_{idx}",
            "value": html.escape(a.label),
            "style": style,
            "edge": "1",
            "parent": "1",
        }
        if a.src_id:
            attrs["source"] = a.src_id
        if a.dst_id:
            attrs["target"] = a.dst_id
        cell = ET.SubElement(root, "mxCell", **attrs)
        geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        ET.SubElement(
            geom,
            "mxPoint",
            x=str(round(a.src_xy[0], 1)),
            y=str(round(a.src_xy[1], 1)),
            **{"as": "sourcePoint"},
        )
        ET.SubElement(
            geom,
            "mxPoint",
            x=str(round(a.dst_xy[0], 1)),
            y=str(round(a.dst_xy[1], 1)),
            **{"as": "targetPoint"},
        )
        if a.waypoints:
            arr = ET.SubElement(geom, "Array", **{"as": "points"})
            for wx, wy in a.waypoints:
                ET.SubElement(arr, "mxPoint", x=str(round(wx, 1)), y=str(round(wy, 1)))

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def _draw_box_text(ax, b: Box, z_text: int) -> None:
    """Place title and subtitle with exact line-count-aware vertical centering."""
    cx = b.x + b.w / 2.0
    cy = b.y + b.h / 2.0
    if not b.subtitle:
        ax.text(
            cx, cy, b.title,
            fontsize=b.title_size, fontweight="bold",
            color=b.text_color, ha="center", va="center",
            linespacing=1.18, zorder=z_text,
        )
        return

    n_title = b.title.count("\n") + 1
    n_sub = b.subtitle.count("\n") + 1
    lh_title = b.title_size * 1.25
    lh_sub = b.sub_size * 1.28
    gap = 4.5
    total_h = n_title * lh_title + gap + n_sub * lh_sub
    y_top = cy - total_h / 2.0
    title_cy = y_top + (n_title * lh_title) / 2.0
    sub_cy = y_top + n_title * lh_title + gap + (n_sub * lh_sub) / 2.0

    ax.text(
        cx, title_cy, b.title,
        fontsize=b.title_size, fontweight="bold",
        color=b.text_color, ha="center", va="center",
        linespacing=1.18, zorder=z_text,
    )
    ax.text(
        cx, sub_cy, b.subtitle,
        fontsize=b.sub_size, color=b.text_color,
        ha="center", va="center", linespacing=1.22, zorder=z_text,
    )


def render_diagram(
    pdf_path: Path,
    png_path: Path,
    width: float,
    height: float,
    boxes: list[Box],
    arrows: list[Arrow],
    legend_items: Optional[list[tuple[float, str, str, str, bool]]] = None,
) -> None:
    """Render crisp vector PDF and 400-DPI PNG matching the .drawio geometry."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), facecolor="#FFFFFF")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="#FFFFFF")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    if legend_items:
        y_leg = 14.0
        for x_cur, label, fill, edge, is_line in legend_items:
            if is_line:
                ax.plot([x_cur, x_cur + 18], [y_leg, y_leg], color=edge, lw=1.5, ls="--", zorder=5)
                ax.text(x_cur + 23, y_leg + 0.5, label, fontsize=8.0, color="#1E293B",
                        va="center", ha="left", zorder=5)
            else:
                patch = FancyBboxPatch(
                    (x_cur, y_leg - 5.5), 14, 11,
                    boxstyle="round,pad=0,rounding_size=2",
                    facecolor=fill, edgecolor=edge, linewidth=1.1, zorder=5,
                )
                ax.add_patch(patch)
                ax.text(x_cur + 19, y_leg + 0.5, label, fontsize=8.0, color="#1E293B",
                        va="center", ha="left", zorder=5)

    for b in sorted(boxes, key=lambda item: (0 if item.is_container else 1)):
        ls = "--" if b.dashed else "-"
        if b.is_diamond:
            cx, cy = b.x + b.w / 2.0, b.y + b.h / 2.0
            pts = [(cx, b.y), (b.x + b.w, cy), (cx, b.y + b.h), (b.x, cy)]
            poly = Polygon(
                pts, closed=True, facecolor=b.fill, edgecolor=b.edge,
                linewidth=b.lw, linestyle=ls, zorder=2,
            )
            ax.add_patch(poly)
            _draw_box_text(ax, b, z_text=3)
            continue

        r_size = 5.5 if b.rounded else 0.0
        z_box = 1 if b.is_container else 2
        patch = FancyBboxPatch(
            (b.x, b.y), b.w, b.h,
            boxstyle=f"round,pad=0,rounding_size={r_size}",
            facecolor=b.fill, edgecolor=b.edge,
            linewidth=b.lw, linestyle=ls, zorder=z_box,
        )
        ax.add_patch(patch)

        if b.is_container:
            ax.text(
                b.x + 10, b.y + 13, b.title,
                fontsize=b.title_size, fontweight="bold",
                color=b.text_color, ha="left", va="center", zorder=z_box + 1,
            )
        else:
            _draw_box_text(ax, b, z_text=z_box + 1)

    for a in arrows:
        ls = "--" if a.dashed else "-"
        pts = [a.src_xy] + list(a.waypoints) + [a.dst_xy]
        if len(pts) > 2:
            xs = [p[0] for p in pts[:-1]]
            ys = [p[1] for p in pts[:-1]]
            ax.plot(xs, ys, color=a.color, linewidth=a.lw, linestyle=ls, zorder=4, solid_capstyle="round")
            p_start, p_end = pts[-2], pts[-1]
        else:
            p_start, p_end = pts[0], pts[1]

        arr = FancyArrowPatch(
            p_start, p_end,
            arrowstyle="-|>,head_length=4.5,head_width=2.8",
            color=a.color, linewidth=a.lw, linestyle=ls,
            shrinkA=0, shrinkB=0, zorder=4,
        )
        ax.add_patch(arr)

        if a.label:
            lx, ly = a.label_xy if a.label_xy is not None else (
                (a.src_xy[0] + a.dst_xy[0]) / 2.0,
                (a.src_xy[1] + a.dst_xy[1]) / 2.0,
            )
            ax.text(
                lx, ly, a.label,
                fontsize=a.label_size, fontweight="bold", color=a.color,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.16", facecolor=a.label_bg, edgecolor="none", alpha=0.95),
            )

    fig.savefig(pdf_path, format="pdf", facecolor="#FFFFFF")
    fig.savefig(png_path, format="png", dpi=400, facecolor="#FFFFFF")
    plt.close(fig)


def build_fig1(out_dir: Path) -> None:
    """Fig. 1: Conceptual model (Base replicated vs. Extended instance-level mechanism)."""
    W, H = 716.0, 415.0
    boxes = [
        Box("base_frame", 8, 30, 240, 376, "Base Model (Replicated)",
            fill=PALETTE["frame_base"], edge=PALETTE["base_edge"], text_color=PALETTE["base_text"],
            dashed=True, is_container=True, lw=1.2, title_size=8.4),
        Box("ext_frame", 268, 30, 440, 376, "Extended Model (Fixed Model: Instance Level)",
            fill=PALETTE["frame_ext"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            dashed=False, is_container=True, lw=1.2, title_size=8.4),

        # Base constructs (Blue)
        Box("b_model", 15, 62, 114, 48, "Model Class", "LR, RF, XGB",
            fill=PALETTE["base_fill"], edge=PALETTE["base_edge"], text_color=PALETTE["base_text"],
            title_size=8.2, sub_size=8.0),
        Box("b_idcs", 15, 134, 114, 52, "Cost-Sensitivity", "IDCS vs. Std.",
            fill=PALETTE["base_fill"], edge=PALETTE["base_edge"], text_color=PALETTE["base_text"],
            title_size=8.1, sub_size=8.0),
        Box("b_imb", 15, 210, 114, 56, "Class Imbalance", "Default rate $\\pi$\n(1% to 30%)",
            fill=PALETTE["base_fill"], edge=PALETTE["base_edge"], text_color=PALETTE["base_text"],
            title_size=8.1, sub_size=8.0),
        Box("b_instab", 150, 128, 90, 64, "Retraining\nInstability", "CoV & SRA\n($d=10$)",
            fill=PALETTE["base_fill"], edge=PALETTE["base_edge"], text_color=PALETTE["base_text"],
            title_size=8.2, sub_size=8.0),
        Box("b_e1", 15, 336, 225, 54, "E1: Variance Decomposition",
            "Refit vs. explainer sampling noise",
            fill=PALETTE["ctrl_fill"], edge=PALETTE["ctrl_edge"], text_color=PALETTE["ctrl_text"],
            title_size=8.2, sub_size=8.0),

        # Extension constructs (Teal) & Controls (Slate)
        Box("e_pert", 278, 58, 106, 52, "Perturbation", "Gaussian $s\\cdot\\mathrm{SD}_j$",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.2, sub_size=8.0),
        Box("e_cross", 422, 58, 114, 52, "Crossings $C$", "Split $C_{\\mathrm{tree}}$, Bin $C_{\\mathrm{bin}}$",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.2, sub_size=8.0),
        Box("e_deltae", 578, 58, 120, 52, "Local Change $\\Delta E$", "Top-$k$ $D_k$, RC flip",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.2, sub_size=8.0),

        Box("e_mods", 404, 134, 142, 56, "Moderators",
            "H4b: Explainer & Model\nH4c: LIME Discret.",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.2, sub_size=8.0),
        Box("e_floor", 578, 136, 120, 52, "Noise Floor $q_i$",
            "$K=10$ repeat seeds",
            fill=PALETTE["ctrl_fill"], edge=PALETTE["ctrl_edge"], text_color=PALETTE["ctrl_text"],
            title_size=8.2, sub_size=8.0),

        Box("e_dp", 278, 206, 120, 50, "Prediction $\\Delta p$", "$|\\Delta p| \\leq \\tau$ (filter)",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.2, sub_size=8.0),

        Box("e_base", 434, 252, 124, 50, "Baselines (vs. H5)",
            "Margin & Tree Var.",
            fill=PALETTE["ctrl_fill"], edge=PALETTE["ctrl_edge"], text_color=PALETTE["ctrl_text"],
            title_size=8.2, sub_size=8.0),

        Box("e_fx", 278, 332, 146, 58, "Ex-Ante Fragility $F(x)$",
            "Structure-only\n($F_{\\mathrm{tree}}, F_{\\mathrm{bin}}$)",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.2, sub_size=8.0),
        Box("e_yi", 578, 332, 120, 58, "Instability $Y_i$",
            "$\\mathbf{1}[\\bar{U}_i \\geq 0.2]$",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.2, sub_size=8.0),
    ]

    arrows = [
        Arrow((129, 86), (195, 128), "b_model", "b_instab", "", color=PALETTE["base_edge"],
              waypoints=[(195, 86)]),
        Arrow((129, 160), (150, 160), "b_idcs", "b_instab", "H1b", label_xy=(139, 146),
              color=PALETTE["base_edge"]),
        Arrow((129, 238), (195, 192), "b_imb", "b_instab", "H1a", label_xy=(170, 226),
              color=PALETTE["base_edge"], waypoints=[(195, 238)]),
        Arrow((195, 192), (195, 336), "b_instab", "b_e1", "E1", label_xy=(209, 286),
              color=PALETTE["ctrl_edge"], dashed=True),

        Arrow((384, 84), (422, 84), "e_pert", "e_cross", "H4a", label_xy=(403, 71),
              color=PALETTE["ext_edge"]),
        Arrow((536, 84), (578, 84), "e_cross", "e_deltae", "H4a", label_xy=(557, 71),
              color=PALETTE["ext_edge"]),

        Arrow((546, 162), (558, 88), "e_mods", "e_deltae", "H4b/c", label_xy=(534, 118),
              color=PALETTE["ext_edge"], waypoints=[(558, 162)]),

        Arrow((638, 136), (638, 110), "e_floor", "e_deltae", "H2, E1", label_xy=(670, 123),
              color=PALETTE["ctrl_edge"]),

        Arrow((331, 110), (331, 206), "e_pert", "e_dp", "", color=PALETTE["ext_edge"], dashed=True),
        Arrow((398, 224), (578, 97), "e_dp", "e_deltae", "H3 (weak)", label_xy=(492, 212),
              color=PALETTE["warn_edge"], dashed=True, waypoints=[(565, 224), (565, 97)]),

        Arrow((424, 361), (578, 361), "e_fx", "e_yi", "H5", label_xy=(501, 348),
              color=PALETTE["ext_edge"]),
        Arrow((558, 277), (638, 332), "e_base", "e_yi", "vs. H5", label_xy=(604, 277),
              color=PALETTE["ctrl_edge"], dashed=True, waypoints=[(638, 277)]),

        Arrow((248, 361), (278, 361), "base_frame", "e_fx", "H6", label_xy=(263, 347),
              color=PALETTE["base_edge"], dashed=True),
    ]

    legend = [
        (14.0, "Base (Replicated)", PALETTE["base_fill"], PALETTE["base_edge"], False),
        (165.0, "Extended (Proposed)", PALETTE["ext_fill"], PALETTE["ext_edge"], False),
        (332.0, "Control / Baseline", PALETTE["ctrl_fill"], PALETTE["ctrl_edge"], False),
        (486.0, "Weak / Exploratory Path", "", PALETTE["warn_edge"], True),
    ]

    write_drawio(out_dir / "fig1_conceptual_model.drawio", "Conceptual_Model", W, H, boxes, arrows)
    render_diagram(
        out_dir / "fig1_conceptual_model.pdf",
        out_dir / "fig1_conceptual_model.png",
        W, H, boxes, arrows, legend_items=legend,
    )


def build_fig2(out_dir: Path) -> None:
    """Fig. 2: Methodology and research-design flowchart (3 rows x 2 wide columns)."""
    W, H = 716.0, 308.0
    bw, bh = 326.0, 80.0
    xs = [14.0, 376.0]
    ys = [12.0, 114.0, 216.0]

    boxes = [
        Box("s0", xs[0], ys[0], bw, bh, "1. Data Preparation (01_clean.py)",
            "• HMEQ (5,960), SGCS (1,000), Taiwan (30,000)\n"
            "• Train-only median/mode imputation & scaling\n"
            "• Stratified splits (Part 1: 300 test; Part 2: 70/30)",
            fill=PALETTE["ctrl_fill"], edge=PALETTE["ctrl_edge"], text_color=PALETTE["ctrl_text"],
            title_size=8.4, sub_size=8.0),
        Box("s1", xs[1], ys[0], bw, bh, "2. Part 1 Replication & E1 (02_replicate.py)",
            "• Resample default rate $\\pi \\in \\{1\\dots 30\\}\\%$ ($J$ refits)\n"
            "• LR, RF, XGB + IDCS (cslogit, csboost)\n"
            "• PermSHAP & LIME $\\rightarrow$ CoV, SRA (H1a/b, E1)",
            fill=PALETTE["base_fill"], edge=PALETTE["base_edge"], text_color=PALETTE["base_text"],
            title_size=8.4, sub_size=8.0),

        Box("s2", xs[1], ys[1], bw, bh, "3. Fixed-Model Perturbations (03_extend.py)",
            "• Tuned LR, RF, XGB at natural prior ($n=200$/data)\n"
            "• Noise floor ($K=10$) + $R=25$ draws ($s=0.01, 0.05$)\n"
            "• $\\tau$-filter ($|\\Delta p| \\leq 0.05$) & crossings $C_{\\mathrm{tree}}, C_{\\mathrm{bin}}$",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.4, sub_size=8.0),
        Box("s3", xs[0], ys[1], bw, bh, "4. Ex-Ante Fragility & Metrics (03_extend.py)",
            "• Structure-only $F_{\\mathrm{tree}}(x), F_{\\mathrm{bin}}(x)$ (no draws)\n"
            "• Baselines: Decision Margin & RF Tree Variance\n"
            "• Outcomes: $\\Delta E$ ($D_k, D_\\rho, D_1$), RC flip, label $Y_i$",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.4, sub_size=8.0),

        Box("s4", xs[0], ys[2], bw, bh, "5. Statistical Suite & R1–R8 (04_robustness.py)",
            "• Confirmatory H1–H5 + Exploratory H6 (Holm corr.)\n"
            "• 5,000-cluster bootstrap mediation & DeLong test\n"
            "• Ablation R1 & sensitivity checks R2–R8",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.4, sub_size=8.0),
        Box("s5", xs[1], ys[2], bw, bh, "6. Case Study & Audit Gate (run_demo.py)",
            "• HMEQ adverse-action applicant walkthrough\n"
            "• Gate policy: $F(x) > \\tau_F \\rightarrow$ Manual review\n"
            "• Vector PDF & 400-DPI figure export (05_plots.py)",
            fill=PALETTE["amber_fill"], edge=PALETTE["amber_edge"], text_color=PALETTE["amber_text"],
            title_size=8.4, sub_size=8.0),
    ]

    arrows = [
        Arrow((xs[0] + bw, ys[0] + bh / 2), (xs[1], ys[0] + bh / 2), "s0", "s1",
              color=PALETTE["ctrl_edge"]),
        Arrow((xs[1] + bw / 2, ys[0] + bh), (xs[1] + bw / 2, ys[1]), "s1", "s2",
              color=PALETTE["base_edge"]),
        Arrow((xs[1], ys[1] + bh / 2), (xs[0] + bw, ys[1] + bh / 2), "s2", "s3",
              color=PALETTE["ext_edge"]),
        Arrow((xs[0] + bw / 2, ys[1] + bh), (xs[0] + bw / 2, ys[2]), "s3", "s4",
              color=PALETTE["ext_edge"]),
        Arrow((xs[0] + bw, ys[2] + bh / 2), (xs[1], ys[2] + bh / 2), "s4", "s5",
              color=PALETTE["amber_edge"]),
    ]

    write_drawio(out_dir / "fig2_methodology_flowchart.drawio", "Methodology_Flowchart", W, H, boxes, arrows)
    render_diagram(
        out_dir / "fig2_methodology_flowchart.pdf",
        out_dir / "fig2_methodology_flowchart.png",
        W, H, boxes, arrows,
    )


def build_fig8(out_dir: Path) -> None:
    """Fig. 8: Adverse-action operational workflow with ex-ante fragility gate F(x)."""
    W, H = 716.0, 240.0
    boxes = [
        Box("c1", 10, 76, 122, 76, "Rejected\nApplicant",
            "HMEQ features $x$\n(DEBTINC, CLAGE)",
            fill=PALETTE["ctrl_fill"], edge=PALETTE["ctrl_edge"], text_color=PALETTE["ctrl_text"],
            title_size=8.3, sub_size=8.0),
        Box("c2", 154, 76, 118, 76, "Scoring Model $f$",
            "XGBoost / RF\n$p(x) \\geq t \\Rightarrow$ Reject",
            fill=PALETTE["base_fill"], edge=PALETTE["base_edge"], text_color=PALETTE["base_text"],
            title_size=8.3, sub_size=8.0),
        Box("c3", 294, 76, 120, 76, "Local Explainer",
            "TreeSHAP / LIME\nTop-4 $T_4(x)$",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.3, sub_size=8.0),
        Box("c4", 436, 62, 112, 104, "Fragility\nGate",
            "$F(x) > \\tau_F$ ?",
            fill=PALETTE["amber_fill"], edge=PALETTE["amber_edge"], text_color=PALETTE["amber_text"],
            is_diamond=True, lw=1.35, title_size=8.2, sub_size=8.0),
        Box("c5", 576, 20, 130, 78, "Issue Notice",
            "Standard top-4\n(Stable reasons)",
            fill=PALETTE["ext_fill"], edge=PALETTE["ext_edge"], text_color=PALETTE["ext_text"],
            title_size=8.3, sub_size=8.0),
        Box("c6", 574, 134, 134, 86, "Manual Review /\nStabilized List",
            "Flag high-fragility;\nlocal consensus",
            fill=PALETTE["warn_fill"], edge=PALETTE["warn_edge"], text_color=PALETTE["warn_text"],
            title_size=8.3, sub_size=8.0),
    ]

    arrows = [
        Arrow((132, 114), (154, 114), "c1", "c2", color=PALETTE["ctrl_edge"]),
        Arrow((272, 114), (294, 114), "c2", "c3", color=PALETTE["base_edge"]),
        Arrow((414, 114), (436, 114), "c3", "c4", color=PALETTE["ext_edge"]),
        Arrow((492, 62), (576, 59), "c4", "c5", "No ($\\leq \\tau_F$)", label_xy=(534, 44),
              color=PALETTE["ext_edge"], waypoints=[(492, 59)]),
        Arrow((492, 166), (576, 177), "c4", "c6", "Yes ($> \\tau_F$)", label_xy=(534, 192),
              color=PALETTE["warn_edge"], waypoints=[(492, 177)]),
    ]

    write_drawio(out_dir / "fig8_case_study.drawio", "Adverse_Action_Workflow", W, H, boxes, arrows)
    render_diagram(
        out_dir / "fig8_case_study.pdf",
        out_dir / "fig8_case_study.png",
        W, H, boxes, arrows,
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    diag_dir = root / "diagrams"
    paper_fig_dir = root / "paper" / "figures"
    diag_dir.mkdir(parents=True, exist_ok=True)
    paper_fig_dir.mkdir(parents=True, exist_ok=True)

    build_fig1(diag_dir)
    build_fig2(diag_dir)
    build_fig8(diag_dir)

    for stem in ("fig1_conceptual_model", "fig2_methodology_flowchart", "fig8_case_study"):
        for ext in (".pdf", ".png"):
            src = diag_dir / f"{stem}{ext}"
            dst = paper_fig_dir / f"{stem}{ext}"
            shutil.copy2(src, dst)
            print(f"Exported {src.name} -> {dst}")


if __name__ == "__main__":
    main()
