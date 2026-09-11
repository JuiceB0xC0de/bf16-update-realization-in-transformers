"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

lrs = D["lrs"]
fig = make_subplots(rows=len(lrs), cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    subplot_titles=[f"Learning rate {D['lr_label'][k]}" for k in lrs])
for r_i, k in enumerate(lrs, start=1):
    for g in D["groups"]:
        pts = [r for r in D["rows"] if r["lr"] == k and r["group"] == g["name"]]
        xs, ys, sym, size, txt = [], [], [], [], []
        for r in pts:
            focal = r["condition"] == D["focal"]
            xs.append(r["label"]); ys.append(r["mean"]); sym.append(g["symbol"]); size.append(15 if focal else 11)
            txt.append(f"{r['label']}: two-seed mean")
            for sd, v in zip(r["seeds"], r["per_seed"]):
                xs.append(r["label"]); ys.append(v); sym.append(g["symbol"] + "-open"); size.append(7)
                txt.append(f"{r['label']}: seed {sd}")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=g["label"], legendgroup=g["name"], showlegend=(r_i == 1),
                                 marker=dict(color=g["color"], symbol=sym, size=size, line=dict(width=1.5)),
                                 text=txt, hovertemplate="%{text}<br>Δ NLL %{y:.4f}<extra></extra>"), row=r_i, col=1)
    add_reference_line(fig, y=0, label="same loss as plain rounding" if r_i == 1 else "", row=r_i, col=1)
    add_reference_line(fig, y=0.02, label="+0.02" if r_i == 1 else "", row=r_i, col=1)
    add_reference_line(fig, y=-0.02, label="−0.02" if r_i == 1 else "", row=r_i, col=1)
    fig.update_yaxes(title="Δ held-out NLL vs rne (nats)", row=r_i, col=1)
fig.update_xaxes(title="Writer", row=len(lrs), col=1)
apply_theme(fig, height=300 * len(lrs) + 60)

save_figure_bundle(fig, "paired_vs_rne", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
