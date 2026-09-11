"""Write-model predicted overshoot against stored overshoot on the captured steps."""
import json, os
import plotly.graph_objects as go
from silico_figures import apply_theme, save_figure_bundle

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data.json")))
rows = d["rows"]
lo = min(min(r["stored"], r["predicted"]) for r in rows) - 0.01
hi = max(max(r["stored"], r["predicted"]) for r in rows) + 0.01
fig = go.Figure()
fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="predicted = stored",
                         line=dict(color=d["colors"]["diagonal"], dash="dot", width=2)))
for order in sorted({r["order"] for r in rows}):
    sub = [r for r in rows if r["order"] == order]
    fig.add_trace(go.Scatter(
        x=[r["stored"] for r in sub], y=[r["predicted"] for r in sub], mode="markers+text",
        text=[f"step {r['step']}" for r in sub], textposition="top left" if order == "order 0" else "bottom right",
        name=order, marker=dict(size=10, color=d["colors"][order]),
        hovertemplate="%{text}<br>stored %{x:.4f}<br>predicted %{y:.4f}<br>exact agreement %{customdata:.4%}<extra>" + order + "</extra>",
        customdata=[r["exact_write_agreement"] for r in sub]))
fig.update_xaxes(title_text="Stored overshoot ρ", range=[lo, hi])
fig.update_yaxes(title_text="Write-model predicted overshoot ρ", range=[lo, hi], scaleanchor="x")
apply_theme(fig, height=460)
save_figure_bundle(fig, "predicted_vs_stored", root=os.path.dirname(HERE), data=d,
                   alt="Scatter of write-model predicted overshoot against stored overshoot for the six captured steps of each data order, with the diagonal predicted = stored.")
