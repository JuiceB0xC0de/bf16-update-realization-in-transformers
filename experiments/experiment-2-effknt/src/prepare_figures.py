"""Render the report figures from the analysis JSONs (pod-side, silico-figures).

Usage: uv run --no-sync python src/prepare_figures.py --analysis results/analysis [--root figures]
Writes figures/<name>/{<name>.html, data.json, alt.txt, plot.py} for:
  nll_vs_master        paired held-out NLL difference from fp32 master weights by learning rate (90% t intervals)
  nll_absolute         held-out NLL per condition and learning rate, seed mean with per-seed points
  delivery_length      cumulative length ratio ||R||/||I|| per writer and learning rate
  never_moved          never-moved coordinate fraction per writer and learning rate
  delivery_over_steps  cumulative length ratio against training step at lr 2e-4
  never_moved_family   never-moved fraction per parameter family at lr 2e-4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import textwrap

import plotly.graph_objects as go
from silico_figures import EDITORIAL_8, add_reference_line, apply_theme, save_figure_bundle

ORDER = ["rne", "kahan", "selective", "sr", "fp32_master", "torch_adamw"]
LABEL = {"rne": "rne (baseline)", "kahan": "kahan", "selective": "selective", "sr": "stochastic rounding",
         "fp32_master": "fp32 master", "torch_adamw": "torch AdamW (null)"}
FAM_LABEL = {"attention": "attention", "mlp": "MLP", "embedding": "embedding", "norm": "norm"}


def entity_colors(root):
    path = os.path.join(root, "entity_colors.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    colors = {"kahan": EDITORIAL_8[0], "rne": EDITORIAL_8[1], "fp32_master": EDITORIAL_8[2],
              "selective": EDITORIAL_8[3], "sr": EDITORIAL_8[4], "torch_adamw": EDITORIAL_8[5],
              "untrained": EDITORIAL_8[7]}
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(colors, f, indent=1)
    return colors


def lr_label(k: str) -> str:
    return {"0.0002": "2e-4", "0.0001": "1e-4", "5e-05": "5e-5"}.get(k, k)


PLOT_HEADER = textwrap.dedent('''\
    """Rebuild this figure from the co-located data.json (no experiment imports)."""
    import json, os
    import plotly.graph_objects as go
    from silico_figures import add_reference_line, apply_theme, save_figure_bundle
    HERE = os.path.dirname(os.path.abspath(__file__))
    D = json.load(open(os.path.join(HERE, "data.json")))
    ''')


def write_plot_py(root, name, body):
    with open(os.path.join(root, name, "plot.py"), "w") as f:
        f.write(PLOT_HEADER + textwrap.dedent(body) + f'\nsave_figure_bundle(fig, "{name}", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))\n')


# --------------------------------------------------------------------------- figure builders
def fig_nll_vs_master(verdicts, colors, root):
    lrs = sorted(verdicts["vs_master"].keys(), key=float)
    rows = []
    for k in lrs:
        for c, v in verdicts["vs_master"][k].items():
            rows.append({"lr": lr_label(k), "log10_lr": math.log10(float(k)), "condition": c, "label": LABEL[c],
                         "mean": v["mean"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"], "color": colors[c]})
    conds = [c for c in ORDER if c != "fp32_master" and any(r["condition"] == c for r in rows)]
    offsets = {c: (i - (len(conds) - 1) / 2) * 0.03 for i, c in enumerate(conds)}
    data = {"rows": rows, "offsets": offsets, "band": 0.01,
            "alt": "Dot and interval chart of the paired held-out NLL difference from fp32 master weights at three "
                   "learning rates, one series per writer, with 90% Student-t intervals and a shaded ±0.01 band."}

    def build(D):
        fig = go.Figure()
        for c in conds:
            rs = [r for r in D["rows"] if r["condition"] == c]
            fig.add_trace(go.Scatter(
                x=[r["log10_lr"] + D["offsets"][c] for r in rs], y=[r["mean"] for r in rs],
                error_y=dict(type="data", symmetric=False, array=[r["ci_hi"] - r["mean"] for r in rs],
                             arrayminus=[r["mean"] - r["ci_lo"] for r in rs], thickness=1.2, width=4),
                mode="markers+lines", name=rs[0]["label"], marker=dict(size=9, color=rs[0]["color"]),
                line=dict(color=rs[0]["color"], width=1, dash="dot"),
                hovertemplate="%{text}<br>Δ NLL %{y:.4f}<extra></extra>",
                text=[f"{r['label']} at lr {r['lr']}: [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]" for r in rs]))
        add_reference_line(fig, y=0, label="same loss as fp32 master")
        add_reference_line(fig, y=D["band"], label="+0.01")
        add_reference_line(fig, y=-D["band"], label="−0.01")
        ticks = sorted({r["log10_lr"] for r in D["rows"]})
        fig.update_xaxes(title="Learning rate", tickvals=ticks,
                         ticktext=[lr_label(f"{10 ** t:g}") for t in ticks])
        fig.update_yaxes(title="Held-out NLL − NLL(fp32 master), nats per token")
        apply_theme(fig, height=460)
        return fig

    fig = build(data)
    save_figure_bundle(fig, "nll_vs_master", root=root, data=data, alt=data["alt"])
    write_plot_py(root, "nll_vs_master", '''
    conds = list(D["offsets"].keys())
    fig = go.Figure()
    for c in conds:
        rs = [r for r in D["rows"] if r["condition"] == c]
        fig.add_trace(go.Scatter(x=[r["log10_lr"] + D["offsets"][c] for r in rs], y=[r["mean"] for r in rs],
            error_y=dict(type="data", symmetric=False, array=[r["ci_hi"] - r["mean"] for r in rs],
                         arrayminus=[r["mean"] - r["ci_lo"] for r in rs], thickness=1.2, width=4),
            mode="markers+lines", name=rs[0]["label"], marker=dict(size=9, color=rs[0]["color"]),
            line=dict(color=rs[0]["color"], width=1, dash="dot")))
    add_reference_line(fig, y=0, label="same loss as fp32 master")
    add_reference_line(fig, y=D["band"], label="+0.01")
    add_reference_line(fig, y=-D["band"], label="−0.01")
    ticks = sorted({r["log10_lr"] for r in D["rows"]})
    fig.update_xaxes(title="Learning rate", tickvals=ticks, ticktext=[f"{10 ** t:g}" for t in ticks])
    fig.update_yaxes(title="Held-out NLL − NLL(fp32 master), nats per token")
    apply_theme(fig, height=460)
    ''')
    return fig


def fig_nll_absolute(verdicts, runs_csv_rows, colors, root):
    lrs = sorted(verdicts["nll_summary"].keys(), key=float)
    series = []
    for c in ORDER:
        pts = [{"lr": lr_label(k), "log10_lr": math.log10(float(k)), "mean": verdicts["nll_summary"][k][c]["mean"],
                "sd": verdicts["nll_summary"][k][c]["sd"]} for k in lrs if c in verdicts["nll_summary"][k]]
        if pts:
            series.append({"condition": c, "label": LABEL[c], "color": colors[c], "points": pts})
    untrained = float(sum(verdicts["untrained_nll_per_seed"].values()) / len(verdicts["untrained_nll_per_seed"]))
    seed_pts = [{"condition": r["condition"], "log10_lr": math.log10(float(r["lr"])), "nll": float(r["eval_nll"]),
                 "seed": int(r["seed"])} for r in runs_csv_rows]
    data = {"series": series, "untrained": untrained, "seed_points": seed_pts,
            "alt": "Line chart of held-out NLL per token against learning rate for six optimizer conditions, "
                   "seed means with faint per-seed points, and the untrained model's NLL as a reference line."}

    fig = go.Figure()
    for s in data["series"]:
        fig.add_trace(go.Scatter(x=[p["log10_lr"] for p in s["points"]], y=[p["mean"] for p in s["points"]],
                                 mode="lines+markers", name=s["label"], line=dict(color=s["color"], width=2),
                                 marker=dict(size=8, color=s["color"]),
                                 hovertemplate=s["label"] + " lr %{text}: %{y:.4f}<extra></extra>",
                                 text=[p["lr"] for p in s["points"]]))
        pts = [p for p in data["seed_points"] if p["condition"] == s["condition"]]
        fig.add_trace(go.Scatter(x=[p["log10_lr"] for p in pts], y=[p["nll"] for p in pts], mode="markers",
                                 marker=dict(size=5, color=s["color"], opacity=0.35), showlegend=False,
                                 hovertemplate=s["label"] + " seed %{text}: %{y:.4f}<extra></extra>",
                                 text=[str(p["seed"]) for p in pts]))
    add_reference_line(fig, y=data["untrained"], label=f"untrained model {data['untrained']:.3f}")
    ticks = sorted({p["log10_lr"] for s in data["series"] for p in s["points"]})
    fig.update_xaxes(title="Learning rate", tickvals=ticks, ticktext=[lr_label(f"{10 ** t:g}") for t in ticks])
    fig.update_yaxes(title="Held-out NLL, nats per token")
    apply_theme(fig, height=460)
    save_figure_bundle(fig, "nll_absolute", root=root, data=data, alt=data["alt"])
    write_plot_py(root, "nll_absolute", '''
    fig = go.Figure()
    for s in D["series"]:
        fig.add_trace(go.Scatter(x=[p["log10_lr"] for p in s["points"]], y=[p["mean"] for p in s["points"]],
                                 mode="lines+markers", name=s["label"], line=dict(color=s["color"], width=2),
                                 marker=dict(size=8, color=s["color"])))
        pts = [p for p in D["seed_points"] if p["condition"] == s["condition"]]
        fig.add_trace(go.Scatter(x=[p["log10_lr"] for p in pts], y=[p["nll"] for p in pts], mode="markers",
                                 marker=dict(size=5, color=s["color"], opacity=0.35), showlegend=False))
    add_reference_line(fig, y=D["untrained"], label=f"untrained model {D['untrained']:.3f}")
    ticks = sorted({p["log10_lr"] for s in D["series"] for p in s["points"]})
    fig.update_xaxes(title="Learning rate", tickvals=ticks, ticktext=[f"{10 ** t:g}" for t in ticks])
    fig.update_yaxes(title="Held-out NLL, nats per token")
    apply_theme(fig, height=460)
    ''')
    return fig


def _grouped_bars(delivery, metric, colors, root, name, ytitle, ref, alt, conds=None, height=420):
    lrs = sorted(delivery.keys(), key=float)
    conds = conds or [c for c in ORDER if any(c in delivery[k] for k in lrs)]
    bars = []
    for c in conds:
        ys, sds = [], []
        for k in lrs:
            e = delivery[k].get(c)
            ys.append(e[metric]["mean"] if e else None)
            sds.append(e[metric]["sd"] if e else None)
        bars.append({"condition": c, "label": LABEL[c], "color": colors[c], "x": [lr_label(k) for k in lrs],
                     "y": ys, "sd": sds})
    data = {"bars": bars, "ref": ref, "alt": alt}
    fig = go.Figure()
    for b in data["bars"]:
        fig.add_trace(go.Bar(x=b["x"], y=b["y"], name=b["label"], marker_color=b["color"],
                             error_y=dict(type="data", array=b["sd"], thickness=1, width=3),
                             hovertemplate=b["label"] + " lr %{x}: %{y:.4f}<extra></extra>"))
    if ref is not None:
        add_reference_line(fig, y=ref["y"], label=ref["label"])
    fig.update_xaxes(title="Learning rate")
    fig.update_yaxes(title=ytitle)
    fig.update_layout(barmode="group")
    apply_theme(fig, height=height)
    save_figure_bundle(fig, name, root=root, data=data, alt=alt)
    write_plot_py(root, name, f'''
    fig = go.Figure()
    for b in D["bars"]:
        fig.add_trace(go.Bar(x=b["x"], y=b["y"], name=b["label"], marker_color=b["color"],
                             error_y=dict(type="data", array=b["sd"], thickness=1, width=3)))
    if D.get("ref"):
        add_reference_line(fig, y=D["ref"]["y"], label=D["ref"]["label"])
    fig.update_xaxes(title="Learning rate")
    fig.update_yaxes(title={ytitle!r})
    fig.update_layout(barmode="group")
    apply_theme(fig, height={height})
    ''')
    return fig


def fig_delivery_over_steps(series, colors, root, lr_key="0.0002"):
    S = series[lr_key]
    traces = [{"condition": c, "label": LABEL[c], "color": colors[c], "steps": S[c]["steps"],
               "y": S[c]["cumulative_length_ratio"]} for c in ORDER if c in S]
    data = {"lr": lr_label(lr_key), "traces": traces,
            "alt": f"Line chart of the cumulative length ratio of realized to intended update against training step "
                   f"at learning rate {lr_label(lr_key)}, one line per writer, seed means, with a reference line at 1."}
    fig = go.Figure()
    for t in data["traces"]:
        fig.add_trace(go.Scatter(x=t["steps"], y=t["y"], mode="lines", name=t["label"],
                                 line=dict(color=t["color"], width=2.2 if t["condition"] == "kahan" else 1.6),
                                 hovertemplate=t["label"] + " step %{x}: %{y:.4f}<extra></extra>"))
    add_reference_line(fig, y=1.0, label="full delivery")
    fig.update_xaxes(title="Training step")
    fig.update_yaxes(title="Cumulative ‖R‖ / ‖I‖")
    apply_theme(fig, height=440)
    save_figure_bundle(fig, "delivery_over_steps", root=root, data=data, alt=data["alt"])
    write_plot_py(root, "delivery_over_steps", '''
    fig = go.Figure()
    for t in D["traces"]:
        fig.add_trace(go.Scatter(x=t["steps"], y=t["y"], mode="lines", name=t["label"],
                                 line=dict(color=t["color"], width=2.2 if t["condition"] == "kahan" else 1.6)))
    add_reference_line(fig, y=1.0, label="full delivery")
    fig.update_xaxes(title="Training step")
    fig.update_yaxes(title="Cumulative ‖R‖ / ‖I‖")
    apply_theme(fig, height=440)
    ''')
    return fig


def fig_never_moved_family(delivery, colors, root, lr_key="0.0002"):
    D = delivery[lr_key]
    fams = ["attention", "mlp", "embedding", "norm"]
    conds = [c for c in ["rne", "kahan", "selective", "sr", "fp32_master"] if c in D]
    bars = [{"condition": c, "label": LABEL[c], "color": colors[c], "x": [FAM_LABEL[f] for f in fams],
             "y": [D[c]["families"][f]["never_moved_fraction"]["mean"] for f in fams],
             "sd": [D[c]["families"][f]["never_moved_fraction"]["sd"] for f in fams]} for c in conds]
    data = {"lr": lr_label(lr_key), "bars": bars,
            "alt": f"Grouped bar chart of the fraction of coordinates that never moved over 3,000 steps at learning "
                   f"rate {lr_label(lr_key)}, by parameter family and writer, seed means with sd error bars."}
    fig = go.Figure()
    for b in bars:
        fig.add_trace(go.Bar(x=b["x"], y=b["y"], name=b["label"], marker_color=b["color"],
                             error_y=dict(type="data", array=b["sd"], thickness=1, width=3),
                             hovertemplate=b["label"] + " %{x}: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title="Parameter family")
    fig.update_yaxes(title="Never-moved fraction", range=[0, 1])
    fig.update_layout(barmode="group")
    apply_theme(fig, height=420)
    save_figure_bundle(fig, "never_moved_family", root=root, data=data, alt=data["alt"])
    write_plot_py(root, "never_moved_family", '''
    fig = go.Figure()
    for b in D["bars"]:
        fig.add_trace(go.Bar(x=b["x"], y=b["y"], name=b["label"], marker_color=b["color"],
                             error_y=dict(type="data", array=b["sd"], thickness=1, width=3)))
    fig.update_xaxes(title="Parameter family")
    fig.update_yaxes(title="Never-moved fraction", range=[0, 1])
    fig.update_layout(barmode="group")
    apply_theme(fig, height=420)
    ''')
    return fig


def main():
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--root", default="figures")
    args = ap.parse_args()
    with open(os.path.join(args.analysis, "verdicts.json")) as f:
        verdicts = json.load(f)
    with open(os.path.join(args.analysis, "delivery_table.json")) as f:
        delivery = json.load(f)
    with open(os.path.join(args.analysis, "step_series.json")) as f:
        series = json.load(f)
    with open(os.path.join(args.analysis, "runs_table.csv")) as f:
        rows = list(csv.DictReader(f))
    colors = entity_colors(args.root)
    fig_nll_vs_master(verdicts, colors, args.root)
    fig_nll_absolute(verdicts, rows, colors, args.root)
    _grouped_bars(delivery, "cumulative_length_ratio", colors, args.root, "delivery_length",
                  "Cumulative ‖R‖ / ‖I‖", {"y": 1.0, "label": "full delivery"},
                  "Grouped bar chart of the cumulative length ratio of realized to intended update over 3,000 steps, "
                  "per writer and learning rate, seed means with sd error bars, and a reference line at 1.")
    _grouped_bars(delivery, "never_moved_fraction", colors, args.root, "never_moved",
                  "Never-moved fraction", None,
                  "Grouped bar chart of the fraction of coordinates whose bf16 value never changed over 3,000 steps, "
                  "per writer and learning rate, seed means with sd error bars.")
    fig_delivery_over_steps(series, colors, args.root)
    fig_never_moved_family(delivery, colors, args.root)
    print("figures written under", args.root)


if __name__ == "__main__":
    main()
