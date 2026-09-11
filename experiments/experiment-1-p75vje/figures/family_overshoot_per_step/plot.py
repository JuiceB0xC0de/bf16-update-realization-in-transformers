"""Pooled bf16 overshoot per step, by parameter family (order 0)."""
import json, os
import pandas as pd
import plotly.express as px
from silico_figures import apply_theme, add_reference_line, save_figure_bundle

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data.json")))
df = pd.DataFrame(d["rows"])
fig = px.line(df, x="step", y="overshoot", color="family", render_mode="svg", color_discrete_map=d["colors"],
              labels={"step": "Training step", "overshoot": "Pooled bf16 overshoot ρ", "family": "Family"})
add_reference_line(fig, y=1.0, label="ρ = 1 (nothing lost)")
fig.update_layout(legend_title_text="")
apply_theme(fig, height=420)
save_figure_bundle(fig, "family_overshoot_per_step", root=os.path.dirname(HERE), data=d,
                   alt="Line chart of the pooled bfloat16 overshoot per training step for the attention, MLP, embedding and norm parameter families of SmolLM2-135M on data order 0; the norm series is noisy because only about 0.4% of its 35,136 coordinates move on a typical step.")
