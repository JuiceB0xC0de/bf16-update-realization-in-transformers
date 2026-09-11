"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

fig = go.Figure()
for g in D["groups"]:
    fig.add_trace(go.Bar(x=[D["fam_label"][f] for f in D["fams"]], y=g["values"], name=g["label"],
                         marker=dict(color=g["color"], opacity=g["opacity"], pattern=dict(shape=g["pattern"])),
                         hovertemplate=g["label"] + "<br>%{x}: %{y:.3f}<extra></extra>"))
fig.update_layout(barmode="group")
fig.update_xaxes(title="Parameter family")
fig.update_yaxes(title="Never-moved coordinate fraction", range=[0, 1])
apply_theme(fig, height=420)

save_figure_bundle(fig, "never_moved_family_25_lr2e-4", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
