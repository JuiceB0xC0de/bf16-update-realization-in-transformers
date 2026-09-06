"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

fig = go.Figure()
for t in D["traces"]:
    fig.add_trace(go.Scatter(x=t["steps"], y=t["y"], mode="lines", name=t["label"],
                             line=dict(color=t["color"], width=2.2 if t["condition"] == "kahan" else 1.6)))
add_reference_line(fig, y=1.0, label="full delivery")
fig.update_xaxes(title="Training step")
fig.update_yaxes(title="Cumulative ‖R‖ / ‖I‖")
apply_theme(fig, height=440)

save_figure_bundle(fig, "delivery_over_steps", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
