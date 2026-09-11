"""Distribution of x = |intended update| / bf16 spacing, per family, against the Mamba reference."""
import json, os
import plotly.graph_objects as go
from silico_figures import apply_theme, add_reference_line, save_figure_bundle

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data.json")))
fig = go.Figure()
for s in d["series"]:
    focal = s["name"] != "Mamba-3 187M reference"
    fig.add_trace(go.Scatter(x=d["log2_bin_lower"], y=s["share"], mode="lines+markers", name=s["name"],
                             line=dict(color=d["colors"][s["name"]], width=2.5 if focal else 1.5,
                                       dash="solid" if focal else "dash", shape="hv"),
                             marker=dict(size=5), opacity=1.0 if focal else 0.8))
add_reference_line(fig, x=-1, label="rounding threshold x = 0.5")
fig.update_xaxes(title_text="log₂(x), x = |intended update| / bf16 spacing", range=d["x_range"])
fig.update_yaxes(title_text="Share of coordinates in bin")
apply_theme(fig, height=420)
save_figure_bundle(fig, "x_histogram", root=os.path.dirname(HERE), data=d,
                   alt="Histogram in log2 bins of the intended update measured in bfloat16 spacings, one series per SmolLM2-135M parameter family at step 100 and one for the Mamba-3 step-100 reference bundle, with the rounding threshold at x = 0.5 marked.")
