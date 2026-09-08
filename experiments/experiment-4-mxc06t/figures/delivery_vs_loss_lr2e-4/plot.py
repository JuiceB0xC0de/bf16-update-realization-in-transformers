"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

lrs = D["lrs"]
fig = go.Figure()
for r_i, k in enumerate(lrs, start=1):
    P = D["points"][k]
    keno = [p for p in P if p["series"] == "keno_drop"]
    keno.sort(key=lambda p: p["x"])
    fig.add_trace(go.Scatter(x=[p["x"] for p in keno], y=[p["y"] for p in keno], mode="lines+markers", name="random pick (keno drop)",
                             legendgroup="keno", showlegend=(r_i == 1), line=dict(color=D["colors"]["keno_drop"], width=2.5),
                             marker=dict(size=[11 if p["n"] == 5 else 8 for p in keno], color=D["colors"]["keno_drop"],
                                         opacity=[1.0 if p["n"] == 5 else 0.55 for p in keno]),
                             error_y=dict(type="data", array=[p["sd"] or 0 for p in keno], thickness=1.2, width=4),
                             text=[p["label"] for p in keno], hovertemplate="%{text}<br>delivery %{x:.3f}<br>NLL %{y:.4f}<extra></extra>"))
    thr = sorted([p for p in P if p["series"] == "thr_drop"], key=lambda p: p["x"])
    fig.add_trace(go.Scatter(x=[p["x"] for p in thr], y=[p["y"] for p in thr], mode="lines+markers", name="magnitude pick (threshold drop)",
                             legendgroup="thr", showlegend=(r_i == 1), line=dict(color=D["colors"]["thr_drop"], width=2.5, dash="dot"),
                             marker=dict(size=11, symbol="diamond", color=D["colors"]["thr_drop"]),
                             error_y=dict(type="data", array=[p["sd"] or 0 for p in thr], thickness=1.2, width=4),
                             text=[p["label"] for p in thr], hovertemplate="%{text}<br>delivery %{x:.3f}<br>NLL %{y:.4f}<extra></extra>"))
    for ref, sym in (("rne", "square"), ("kahan", "x"), ("fp32_master", "triangle-up")):
        pts = [p for p in P if p["series"] == ref]
        if pts:
            p = pts[0]
            fig.add_trace(go.Scatter(x=[p["x"]], y=[p["y"]], mode="markers", name=p["label"], legendgroup=ref, showlegend=(r_i == 1),
                                     marker=dict(size=12, symbol=sym, color=D["colors"][ref], line=dict(width=1.5)),
                                     error_y=dict(type="data", array=[p["sd"] or 0], thickness=1.2, width=4),
                                     hovertemplate=p["label"] + "<br>delivery %{x:.3f}<br>NLL %{y:.4f}<extra></extra>"))
    add_reference_line(fig, y=D["bar"], label="fp32 master at lr 5e-5")
    fig.update_yaxes(title="Held-out NLL (nats/token)")
    fig.update_xaxes(title="Cumulative delivery ‖R‖/‖I‖", range=[0.1, 1.05])
apply_theme(fig, height=440)

save_figure_bundle(fig, "delivery_vs_loss_lr2e-4", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
