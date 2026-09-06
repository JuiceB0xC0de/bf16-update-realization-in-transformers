"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

conds = list(D["offsets"].keys())
fig = go.Figure()
for c in conds:
    rs = [r for r in D["rows"] if r["condition"] == c]
    fig.add_trace(go.Scatter(x=[r["log10_lr"] + D["offsets"][c] for r in rs], y=[r["mean"] for r in rs],
        error_y=dict(type="data", symmetric=False, array=[r["ci_hi"] - r["mean"] for r in rs],
                     arrayminus=[r["mean"] - r["ci_lo"] for r in rs], thickness=1.2, width=4),
        mode="markers+lines", name=rs[0]["label"], marker=dict(size=9, color=rs[0]["color"]),
        line=dict(color=rs[0]["color"], width=1, dash="dot")))
add_reference_line(fig, y=0, label="same loss as fp32 master")
add_reference_line(fig, y=D["band"], label="+0.01")
add_reference_line(fig, y=-D["band"], label="−0.01")
ticks = sorted({r["log10_lr"] for r in D["rows"]})
fig.update_xaxes(title="Learning rate", tickvals=ticks, ticktext=[f"{10 ** t:g}" for t in ticks])
fig.update_yaxes(title="Held-out NLL − NLL(fp32 master), nats per token")
apply_theme(fig, height=460)

save_figure_bundle(fig, "nll_vs_master", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
