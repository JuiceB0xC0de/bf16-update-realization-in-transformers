"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

fig = go.Figure()
for s in D["series"]:
    fig.add_trace(go.Scatter(x=[p["log10_lr"] for p in s["points"]], y=[p["mean"] for p in s["points"]],
                             mode="lines+markers", name=s["label"], line=dict(color=s["color"], width=2),
                             marker=dict(size=8, color=s["color"])))
    pts = [p for p in D["seed_points"] if p["condition"] == s["condition"]]
    fig.add_trace(go.Scatter(x=[p["log10_lr"] for p in pts], y=[p["nll"] for p in pts], mode="markers",
                             marker=dict(size=5, color=[s["color"]] * len(pts), opacity=0.35), showlegend=False,
                             name=s["label"], legendgroup=s["label"]))
add_reference_line(fig, y=D["untrained"], label=f"untrained model {D['untrained']:.3f}")
ticks = sorted({p["log10_lr"] for s in D["series"] for p in s["points"]})
fig.update_xaxes(title="Learning rate", tickvals=ticks, ticktext=[f"{10 ** t:g}" for t in ticks])
fig.update_yaxes(title="Held-out NLL, nats per token")
apply_theme(fig, height=460)

save_figure_bundle(fig, "nll_absolute", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
