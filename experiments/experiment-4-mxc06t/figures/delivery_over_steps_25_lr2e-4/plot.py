"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

fig = go.Figure()
for s in D["series"]:
    fig.add_trace(go.Scatter(x=s["steps"], y=s["y"], mode="lines", name=s["label"],
                             line=dict(color=s["color"], width=2.2 if s["kind"] == "keno_drop" else 2.2, dash=s["dash"]),
                             hovertemplate=s["label"] + "<br>step %{x}<br>delivery %{y:.3f}<extra></extra>"))
fig.update_xaxes(title="Training step")
fig.update_yaxes(title="Cumulative delivery ‖R‖/‖I‖")
apply_theme(fig, height=440)

save_figure_bundle(fig, "delivery_over_steps_25_lr2e-4", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
