"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

fig = go.Figure()
for b in D["bars"]:
    fig.add_trace(go.Bar(x=b["x"], y=b["y"], name=b["label"], marker_color=b["color"],
                         error_y=dict(type="data", array=b["sd"], thickness=1, width=3)))
fig.update_xaxes(title="Parameter family")
fig.update_yaxes(title="Never-moved fraction", range=[0, 1])
fig.update_layout(barmode="group")
apply_theme(fig, height=420)

save_figure_bundle(fig, "never_moved_family", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
