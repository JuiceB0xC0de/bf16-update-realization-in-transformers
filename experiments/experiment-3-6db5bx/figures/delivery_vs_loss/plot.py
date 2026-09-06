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
        pts = [p for p in D["points"] if p["lr"] == k and p["group"] == g["name"]]
        if not pts:
            continue
        xs, ys, sym, size, txt, lab = [], [], [], [], [], []
        for p in pts:
            focal = p["condition"] == D["focal"]
            xs.append(p["x"]); ys.append(p["y"]); sym.append(g["symbol"]); size.append(15 if focal else 11)
            txt.append(f"{p['label']}: two-seed mean"); lab.append(p["short"])
            for xv, yv in zip(p["x_per_seed"], p["y_per_seed"]):
                xs.append(xv); ys.append(yv); sym.append(g["symbol"] + "-open"); size.append(6); txt.append(p["label"] + ": one seed"); lab.append("")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=lab, textposition=g["textpos"],
                                 marker=dict(color=g["color"], symbol=sym, size=size, line=dict(width=1.2)),
                                 name=g["label"], legendgroup=g["name"], showlegend=(r_i == 1),
                                 customdata=txt, hovertemplate="%{customdata}<br>delivery %{x:.3f}, NLL %{y:.4f}<extra></extra>"),
                      row=r_i, col=1)
    add_reference_line(fig, y=D["bar"], label=f"{D['bar']:.3f}: fp32 master at lr 5e-5" if r_i == 1 else "", row=r_i, col=1)
    fig.update_yaxes(title="Held-out NLL (nats per token)", row=r_i, col=1)
fig.update_xaxes(title="Cumulative delivery ‖R‖/‖I‖ (fraction of the request written)", range=[0, 1.08], row=len(lrs), col=1)
apply_theme(fig, height=340 * len(lrs) + 60)

save_figure_bundle(fig, "delivery_vs_loss", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
