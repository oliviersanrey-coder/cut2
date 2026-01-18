import io
import csv
import streamlit as st
import matplotlib.pyplot as plt

import optimizer
from optimizer import PanelSpec, PieceType, solve

st.set_page_config(page_title="Cut Optimizer", layout="wide")

st.title("Wood cut optimizer (simple)")
st.caption(f"Optimizer version: {optimizer.OPTIMIZER_VERSION}")

def draw_layout_png(panel: PanelSpec, layout) -> bytes:
    UL, UW = panel.usable_L, panel.usable_W

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_patch(plt.Rectangle((0, 0), UL, UW, fill=False, linewidth=2))
    ax.set_aspect("equal", adjustable="box")

    for pl in layout.placements:
        ax.add_patch(plt.Rectangle((pl.x, pl.y), pl.L, pl.W, fill=False, linewidth=1))
        suffix = " (rot)" if pl.rotated else ""
        label = f"{pl.piece_name}{suffix}\n{pl.L}x{pl.W} mm"
        ax.text(pl.x + pl.L / 2, pl.y + pl.W / 2, label, ha="center", va="center", fontsize=7)

    ax.set_xlim(0, UL)
    ax.set_ylim(0, UW)
    ax.set_xlabel("Length (mm)")
    ax.set_ylabel("Width (mm)")
    ax.set_title(f"{layout.strategy} - usable {UL} x {UW} mm")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def layout_to_csv_bytes(panel: PanelSpec, layout, panel_index: int) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)

    w.writerow(["Panel", panel_index])
    w.writerow(["Strategy", layout.strategy])
    w.writerow(["Usable_L_mm", panel.usable_L, "Usable_W_mm", panel.usable_W, "Kerf_mm", panel.kerf])
    w.writerow([])
    w.writerow(["Note", "rotated means input length/width swapped (only possible when grain_dim=0)."])
    w.writerow([])
    w.writerow(["Piece", "Piece_L_mm", "Piece_W_mm", "Rotated", "x_mm", "y_mm"])

    for p in layout.placements:
        w.writerow([p.piece_name, p.L, p.W, "Yes" if p.rotated else "No", p.x, p.y])

    return buf.getvalue().encode("utf-8")

with st.sidebar:
    st.header("Panel")
    panel_L = st.number_input("Panel length L (mm)", min_value=1, value=2480, step=10)
    panel_W = st.number_input("Panel width W (mm)", min_value=1, value=1200, step=10)
    trim_L = st.number_input("Trim on LENGTH each side (mm)", min_value=0, value=0, step=1)
    trim_W = st.number_input("Trim on WIDTH each side (mm)", min_value=0, value=0, step=1)
    kerf = st.number_input("Kerf (mm)", min_value=0, value=3, step=1)

    st.header("Pieces")
    st.caption("Grain: L (no rotation), W (force rotation), 0 (rotation allowed)")

    if "pieces" not in st.session_state:
        st.session_state.pieces = [
            {"name": "top", "L": 1600, "W": 400, "qty": 2, "grain": "L"},
            {"name": "cote", "L": 650, "W": 400, "qty": 2, "grain": "L"},
            {"name": "inside", "L": 410, "W": 300, "qty": 4, "grain": "0"},
        ]

    for idx, p in enumerate(st.session_state.pieces):
        st.subheader(f"Piece type {idx+1}")
        p["name"] = st.text_input("Name", value=p["name"], key=f"name_{idx}")
        p["L"] = st.number_input("Length (mm)", min_value=1, value=int(p["L"]), step=1, key=f"L_{idx}")
        p["W"] = st.number_input("Width (mm)", min_value=1, value=int(p["W"]), step=1, key=f"W_{idx}")
        p["qty"] = st.number_input("Qty", min_value=1, value=int(p["qty"]), step=1, key=f"qty_{idx}")
        p["grain"] = st.selectbox("Grain (L/W/0)", options=["L", "W", "0"], index=["L","W","0"].index(p["grain"]), key=f"grain_{idx}")
        st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Add piece type"):
            st.session_state.pieces.append({"name": "piece", "L": 100, "W": 100, "qty": 1, "grain": "0"})
            st.rerun()
    with col_b:
        if st.button("Remove last"):
            if len(st.session_state.pieces) > 1:
                st.session_state.pieces.pop()
                st.rerun()

    run = st.button("Compute best solution")

if not run:
    st.info("Set panel + pieces on the left, then click Compute best solution.")
    st.stop()

panel = PanelSpec(
    L=int(panel_L),
    W=int(panel_W),
    trim_L_each_side=int(trim_L),
    trim_W_each_side=int(trim_W),
    kerf=int(kerf)
)

piece_specs = [
    PieceType(
        base_name=p["name"].strip() or f"piece{i+1}",
        length=int(p["L"]),
        width=int(p["W"]),
        qty=int(p["qty"]),
        grain_dim=p["grain"]
    )
    for i, p in enumerate(st.session_state.pieces)
]

solutions = solve(panel, piece_specs)

st.write(f"Usable panel: {panel.usable_L} x {panel.usable_W} mm")
st.write(f"Usable panel area: {panel.usable_area_mm2/1_000_000:.3f} m²")

for s_idx, sol in enumerate(solutions, start=1):
    used_m2 = sol.used_area_mm2 / 1_000_000
    total_m2 = sol.panels_used * (panel.usable_area_mm2 / 1_000_000)

    st.subheader(f"Solution {s_idx}: {sol.strategy}")
    st.write(f"Panels needed: {sol.panels_used}")
    st.write(f"Surface used: {used_m2:.3f} m² out of {total_m2:.3f} m² ({sol.utilization_pct:.1f}%)")
    st.write(f"Estimated cuts: {sol.est_cuts}")
    st.caption("Note: 'rot' means L/W swapped (only possible when grain=0). Labels show actual cut size.")

    for p_idx, layout in enumerate(sol.panel_layouts, start=1):
        png_bytes = draw_layout_png(panel, layout)
        st.image(png_bytes, caption=f"{layout.strategy} - panel {p_idx}", use_container_width=True)

        file_base = f"cutting_{sol.strategy.lower()}_panel{p_idx}"
        st.download_button(
            label=f"Download PNG (panel {p_idx})",
            data=png_bytes,
            file_name=f"{file_base}.png",
            mime="image/png"
        )

        csv_bytes = layout_to_csv_bytes(panel, layout, p_idx)
        st.download_button(
            label=f"Download CSV (panel {p_idx})",
            data=csv_bytes,
            file_name=f"{file_base}.csv",
            mime="text/csv"
        )

        st.divider()
