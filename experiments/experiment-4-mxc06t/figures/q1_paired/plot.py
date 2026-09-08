"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

rows = D["rows"]
fig = go.Figure()
xs = [r["label"] for r in rows]
# per-seed points (open, small)
px_, py_, pt_ = [], [], []
for r in rows:
    for s, v in zip(r["seeds"], r["per_seed"]):
        px_.append(r["label"]); py_.append(v); pt_.append(f"{r['label']}: seed {s}")
fig.add_trace(go.Scatter(x=px_, y=py_, mode="markers", name="per-seed difference",
                         marker=dict(color=D["color_seed"], symbol="circle-open", size=7, line=dict(width=1.2)),
                         text=pt_, hovertemplate="%{text}<br>Δ NLL %{y:.4f}<extra></extra>"))
for kind, sym, size, nm in (("primary", "diamond", 14, "pre-registered primary contrast"), ("secondary", "circle", 11, "secondary contrast")):
    sub = [r for r in rows if r["kind"] == kind]
    if not sub:
        continue
    col = D["color"] if kind == "primary" else D["color_secondary"]
    fig.add_trace(go.Scatter(x=[r["label"] for r in sub], y=[r["mean"] for r in sub], mode="markers", name=nm,
                             marker=dict(color=col, symbol=sym, size=size, line=dict(width=1.5, color=col)),
                             error_y=dict(type="data", symmetric=False, array=[r["hi"] - r["mean"] for r in sub],
                                          arrayminus=[r["mean"] - r["lo"] for r in sub], thickness=2.2, width=8, color=col),
                             text=[f"{r['label']}: {r['verdict']}" for r in sub],
                             hovertemplate="%{text}<br>mean Δ NLL %{y:.4f}<extra></extra>"))
add_reference_line(fig, y=0, label="same loss")
add_reference_line(fig, y=0.01, label="+0.01 band")
add_reference_line(fig, y=-0.01, label="−0.01 band")
fig.update_xaxes(title="Matched level @ learning rate", categoryorder="array", categoryarray=xs)
fig.update_yaxes(title="Δ held-out NLL, random − magnitude (nats)")
apply_theme(fig, height=440)

save_figure_bundle(fig, "q1_paired", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
