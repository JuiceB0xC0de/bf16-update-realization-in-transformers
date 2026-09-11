"""Pooled bf16 overshoot per training step, both data orders, against the Mamba median."""
import json, os
import pandas as pd
import plotly.express as px
from silico_figures import apply_theme, add_reference_line, save_figure_bundle

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data.json")))
df = pd.DataFrame(d["rows"])
colors = d["colors"]
fig = px.line(df, x="step", y="overshoot", color="order", render_mode="svg",
              color_discrete_map=colors, labels={"step": "Training step", "overshoot": "Pooled bf16 overshoot ρ", "order": "Data order"})
add_reference_line(fig, y=1.0, label="ρ = 1 (nothing lost)")
add_reference_line(fig, y=d["mamba_median"], label=f"Mamba-3 187M median {d['mamba_median']:.3f}")
fig.update_layout(legend_title_text="")
apply_theme(fig, height=420)
save_figure_bundle(fig, "overshoot_per_step", root=os.path.dirname(HERE), data=d,
                   alt="Line chart of the pooled bfloat16 overshoot of SmolLM2-135M at each of 300 AdamW steps for two data orders, with reference lines at 1.0 and at the Mamba-3 median 1.334.")
