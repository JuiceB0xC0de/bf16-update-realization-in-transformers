"""Rebuild this figure from the co-located data.json (no experiment imports)."""
import json, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import add_reference_line, apply_theme, save_figure_bundle
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data.json")))

fig = go.Figure(go.Heatmap(z=D["z"], x=D["families"], y=D["writers"], zmin=0, zmax=1, colorscale=D["colorscale"],
                           colorbar=dict(title="Never-moved fraction"), text=[[f"{v:.2f}" for v in row] for row in D["z"]],
                           texttemplate="%{text}", hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>"))
fig.update_xaxes(title="Parameter family")
fig.update_yaxes(title="Writer", autorange="reversed")
apply_theme(fig, height=460)

save_figure_bundle(fig, "never_moved_family", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))
