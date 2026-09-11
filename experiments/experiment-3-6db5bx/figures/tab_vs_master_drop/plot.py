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
    for s in D["series"]:
        rows = [r for r in D["rows"] if r["lr"] == k and r["series"] == s["name"]]
        xs, ys, sym, size, txt = [], [], [], [], []
        for r in rows:
            xs.append(r["x"]); ys.append(r["mean"]); sym.append(s["symbol"]); size.append(11); txt.append(f"{s['label']} at {r['x']}: two-seed mean")
            for v in r["per_seed"]:
                xs.append(r["x"]); ys.append(v); sym.append(s["symbol"] + "-open"); size.append(7); txt.append(f"{s['label']} at {r['x']}: one seed")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=s["color"], symbol=sym, size=size, line=dict(width=1.5)),
                                 name=s["label"], legendgroup=s["name"], showlegend=(r_i == 1),
                                 text=txt, hovertemplate="%{text}<br>Δ NLL %{y:.4f}<extra></extra>"), row=r_i, col=1)
    add_reference_line(fig, y=0, label="no difference" if r_i == 1 else "", row=r_i, col=1)
    add_reference_line(fig, y=0.01, label="+0.01" if r_i == 1 else "", row=r_i, col=1)
    add_reference_line(fig, y=-0.01, label="−0.01" if r_i == 1 else "", row=r_i, col=1)
    fig.update_yaxes(title="Δ held-out NLL (nats)", row=r_i, col=1)
fig.update_xaxes(title="Draw fraction", row=len(lrs), col=1)
apply_theme(fig, height=300 * len(lrs) + 60)

save_figure_bundle(fig, "tab_vs_master_drop", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
