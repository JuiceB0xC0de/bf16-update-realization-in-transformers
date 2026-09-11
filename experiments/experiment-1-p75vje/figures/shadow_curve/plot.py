"""Shadow curve: median pooled overshoot when the reconstructed intent is rescaled by nine factors."""
import json, os
import plotly.graph_objects as go
from silico_figures import apply_theme, add_reference_line, save_figure_bundle

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data.json")))
fig = go.Figure()
for s in d["series"]:
    focal = s["name"].startswith("order")
    fig.add_trace(go.Scatter(x=d["factors"], y=s["overshoot"], mode="lines+markers", name=s["name"],
                             line=dict(color=d["colors"][s["name"]], width=2.5 if focal else 1.5,
                                       dash="solid" if focal else "dash"),
                             marker=dict(size=7 if focal else 5), opacity=1.0 if focal else 0.75))
add_reference_line(fig, y=1.0, label="ρ = 1 (nothing lost)")
fig.update_xaxes(type="log", title_text="Intent scale factor (log)", tickvals=d["factors"],
                 ticktext=[f"{f:g}" if f >= 1 else f"1/{round(1/f)}" for f in d["factors"]])
fig.update_yaxes(title_text="Pooled bf16 overshoot ρ")
apply_theme(fig, height=420)
save_figure_bundle(fig, "shadow_curve", root=os.path.dirname(HERE), data=d,
                   alt="Line chart of the median pooled overshoot against the intent scale factor from 1/16 to 16 on a log axis, for both SmolLM2-135M data orders and the Mamba-3 step-100 reference bundle.")
