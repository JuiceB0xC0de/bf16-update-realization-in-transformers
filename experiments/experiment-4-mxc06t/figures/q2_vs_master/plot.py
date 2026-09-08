"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

rows = D["rows"]
fig = go.Figure()
for series, sym in (("keno_drop", "circle"), ("thr_drop", "diamond")):
    sub = [r for r in rows if r["series"] == series]
    fig.add_trace(go.Scatter(x=[r["label"] for r in sub], y=[r["mean"] for r in sub], mode="markers", name=D["names"][series],
                             marker=dict(color=D["colors"][series], symbol=sym, size=[14 if r["primary"] else 10 for r in sub],
                                         line=dict(width=[2.5 if r["primary"] else 1 for r in sub], color=D["colors"][series])),
                             error_y=dict(type="data", symmetric=False, array=[r["hi"] - r["mean"] for r in sub],
                                          arrayminus=[r["mean"] - r["lo"] for r in sub], thickness=2, width=7, color=D["colors"][series]),
                             text=[f"{r['label']}: {r['verdict']}" + (" (primary)" if r["primary"] else "") for r in sub],
                             hovertemplate="%{text}<br>mean Δ NLL %{y:.4f}<extra></extra>"))
add_reference_line(fig, y=0, label="fp32 master at lr 5e-5")
add_reference_line(fig, y=0.01, label="+0.01 band")
add_reference_line(fig, y=-0.01, label="−0.01 band")
fig.update_xaxes(title="Configuration", categoryorder="array", categoryarray=[r["label"] for r in rows])
fig.update_yaxes(title="Δ held-out NLL vs fp32 master @ 5e-5 (nats)")
apply_theme(fig, height=460)

save_figure_bundle(fig, "q2_vs_master", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
