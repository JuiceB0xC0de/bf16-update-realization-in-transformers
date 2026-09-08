"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                    subplot_titles=[c["title"] for c in D["cells"]])
for ci, cell in enumerate(D["cells"], start=1):
    for r in cell["runs"]:
        if r["divergent"]:
            fig.add_trace(go.Scatter(x=r["steps"], y=r["grad_norm_block_max"], mode="lines+markers", name=r["label"],
                                     line=dict(color=D["color_divergent"][ci - 1], width=2.4, dash="solid" if ci == 1 else "dot"),
                                     marker=dict(size=5, color=D["color_divergent"][ci - 1], symbol="circle" if ci == 1 else "diamond"),
                                     hovertemplate=r["label"] + "<br>steps %{x}<br>max grad norm %{y:.1f}<extra></extra>"), row=1, col=ci)
    others = [r for r in cell["runs"] if not r["divergent"]]
    xs, ys, tx = [], [], []
    for r in others:
        xs += r["steps"] + [None]; ys += r["grad_norm_block_max"] + [None]; tx += [r["label"]] * len(r["steps"]) + [""]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"stable seeds, {cell['short']}", text=tx, connectgaps=False,
                             line=dict(color=D["color_other"][ci - 1], width=1.1, dash=D["dash_other"][ci - 1]), opacity=0.7,
                             hovertemplate="%{text}<br>steps %{x}<br>max grad norm %{y:.1f}<extra></extra>"), row=1, col=ci)
    fig.update_xaxes(title="Training step (100-step block start)", row=1, col=ci)
fig.update_yaxes(type="log", title="Max pre-clip gradient norm in block (log)", row=1, col=1)
apply_theme(fig, height=420)

save_figure_bundle(fig, "instability_primary_thr", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
