"""Render the report figures from results/analysis (pod-side, silico-figures).

Usage: uv run --no-sync python src/prepare_figures.py --analysis results/analysis [--root figures]
Writes figures/<name>/{<name>.html, data.json, alt.txt, plot.py}:
  q1_paired            keno_drop - thr_drop paired held-out NLL difference per level and lr, mean with 90% t interval + per-seed points
  delivery_vs_loss     held-out NLL against cumulative delivery per lr: keno curve, threshold points, plain rounding, kahan, fp32 master
  q2_vs_master         every five-seed keno / threshold configuration minus fp32 master at lr 5e-5, 90% intervals
  never_moved_family   never-moved coordinate fraction per parameter family, keno versus threshold at each level (lr 2e-4)
  delivery_over_steps  cumulative delivery against training step for each keno / threshold pair (lr 2e-4)
Each bundle's plot.py is the exact code used here (the body string is exec'd to build and written to disk).
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import EDITORIAL_8, GRAPHITE, add_reference_line, apply_theme, save_figure_bundle

LEVELS = ["15", "25", "50"]
FAMS = ["attention", "mlp", "embedding", "norm"]
FAM_LABEL = {"attention": "attention", "mlp": "MLP", "embedding": "embedding", "norm": "norm"}
LR_LABEL = {"0.0002": "2e-4", "0.0001": "1e-4", "5e-05": "5e-5"}
LABEL = {"rne": "plain rounding (rne)", "fp32_master": "fp32 master", "kahan": "kahan (exact, all coordinates)"}
PREV_COLORS = {"kahan": "#C4650D", "rne": "#4E728A", "fp32_master": "#2E6E4E", "keno_drop": "#B9605B", "keno_tab": "#31362E",
               "torch_adamw": "#7495AB"}


def entity_colors(root):
    """Continue the previous experiments' mapping; thr_drop takes the next unused EDITORIAL_8 slot."""
    path = os.path.join(root, "entity_colors.json")
    if os.path.exists(path):
        with open(path) as f:
            colors = json.load(f)
    else:
        colors = dict(PREV_COLORS)
    if "thr_drop" not in colors:
        used = set(colors.values())
        colors["thr_drop"] = next(c for c in EDITORIAL_8 if c not in used)
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(colors, f, indent=1)
    return colors


HEADER = textwrap.dedent('''\
    """Rebuild this figure from the co-located data.json (no experiment imports)."""
    import json, os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from silico_figures import add_reference_line, apply_theme, save_figure_bundle
    HERE = os.path.dirname(os.path.abspath(__file__))
    D = json.load(open(os.path.join(HERE, "data.json")))
    ''')


def render(name, body, data, root):
    ns = {"go": go, "make_subplots": make_subplots, "add_reference_line": add_reference_line, "apply_theme": apply_theme, "D": data}
    exec(textwrap.dedent(body), ns)
    fig = ns["fig"]
    save_figure_bundle(fig, name, root=root, data=data, alt=data["alt"])
    with open(os.path.join(root, name, "plot.py"), "w") as f:
        f.write(HEADER + textwrap.dedent(body) +
                f'\nsave_figure_bundle(fig, "{name}", root=os.path.dirname(HERE), data=D, alt=D.get("alt"))\n')
    return fig


# --------------------------------------------------------------------------- 1. Q1 paired intervals
BODY_Q1 = """
rows = D["rows"]
fig = go.Figure()
xs = [r["label"] for r in rows]
# per-seed points (open, small)
px_, py_, pt_ = [], [], []
for r in rows:
    for s, v in zip(r["seeds"], r["per_seed"]):
        px_.append(r["label"]); py_.append(v); pt_.append(f"{r['label']}: seed {s}")
fig.add_trace(go.Scatter(x=px_, y=py_, mode="markers", name="per-seed difference",
                         marker=dict(color=D["color_seed"], symbol="circle-open", size=7, line=dict(width=1.2)),
                         text=pt_, hovertemplate="%{text}<br>Δ NLL %{y:.4f}<extra></extra>"))
for kind, sym, size, nm in (("primary", "diamond", 14, "pre-registered primary contrast"), ("secondary", "circle", 11, "secondary contrast")):
    sub = [r for r in rows if r["kind"] == kind]
    if not sub:
        continue
    col = D["color"] if kind == "primary" else D["color_secondary"]
    fig.add_trace(go.Scatter(x=[r["label"] for r in sub], y=[r["mean"] for r in sub], mode="markers", name=nm,
                             marker=dict(color=col, symbol=sym, size=size, line=dict(width=1.5, color=col)),
                             error_y=dict(type="data", symmetric=False, array=[r["hi"] - r["mean"] for r in sub],
                                          arrayminus=[r["mean"] - r["lo"] for r in sub], thickness=2.2, width=8, color=col),
                             text=[f"{r['label']}: {r['verdict']}" for r in sub],
                             hovertemplate="%{text}<br>mean Δ NLL %{y:.4f}<extra></extra>"))
add_reference_line(fig, y=0, label="same loss")
add_reference_line(fig, y=0.01, label="+0.01 band")
add_reference_line(fig, y=-0.01, label="−0.01 band")
fig.update_xaxes(title="Matched level @ learning rate", categoryorder="array", categoryarray=xs)
fig.update_yaxes(title="Δ held-out NLL, random − magnitude (nats)")
apply_theme(fig, height=440)
"""


def fig_q1(verdicts, colors, root):
    q1 = verdicts["q1_random_vs_magnitude"]
    rows = []
    for kind in ("primary", "secondary"):
        for key, iv in q1[kind].items():
            lv, lrk = key.split("@")
            if iv["lo"] is None:
                continue
            rows.append({"key": key, "label": f"{lv}% @ {LR_LABEL[lrk]}", "kind": kind, "mean": iv["mean"], "lo": iv["lo"], "hi": iv["hi"],
                         "per_seed": iv["per_seed"], "seeds": iv["seeds"], "verdict": iv["verdict"], "sd": iv["sd"]})
    order = {("0.0002", "15"): 0, ("0.0002", "25"): 1, ("0.0002", "50"): 2, ("0.0001", "15"): 3, ("0.0001", "25"): 4, ("0.0001", "50"): 5}
    rows.sort(key=lambda r: order[(r["key"].split("@")[1], r["key"].split("@")[0])])
    data = {"rows": rows, "color": colors["keno_drop"], "color_secondary": EDITORIAL_8[5], "color_seed": GRAPHITE,
            "alt": "Dot-and-interval chart of the paired held-out NLL difference random minus magnitude-based selection at matched "
                   "delivery for three levels and two learning rates; diamonds mark the two pre-registered contrasts, bars are 90% "
                   "Student-t intervals over five seeds, open circles are per-seed values, with reference lines at 0 and ±0.01."}
    return render("q1_paired", BODY_Q1, data, root)


# --------------------------------------------------------------------------- 2. delivery vs loss
BODY_DVL = """
lrs = D["lrs"]
fig = go.Figure()
for r_i, k in enumerate(lrs, start=1):
    P = D["points"][k]
    keno = [p for p in P if p["series"] == "keno_drop"]
    keno.sort(key=lambda p: p["x"])
    fig.add_trace(go.Scatter(x=[p["x"] for p in keno], y=[p["y"] for p in keno], mode="lines+markers", name="random pick (keno drop)",
                             legendgroup="keno", showlegend=(r_i == 1), line=dict(color=D["colors"]["keno_drop"], width=2.5),
                             marker=dict(size=[11 if p["n"] == 5 else 8 for p in keno], color=D["colors"]["keno_drop"],
                                         opacity=[1.0 if p["n"] == 5 else 0.55 for p in keno]),
                             error_y=dict(type="data", array=[p["sd"] or 0 for p in keno], thickness=1.2, width=4),
                             text=[p["label"] for p in keno], hovertemplate="%{text}<br>delivery %{x:.3f}<br>NLL %{y:.4f}<extra></extra>"))
    thr = sorted([p for p in P if p["series"] == "thr_drop"], key=lambda p: p["x"])
    fig.add_trace(go.Scatter(x=[p["x"] for p in thr], y=[p["y"] for p in thr], mode="lines+markers", name="magnitude pick (threshold drop)",
                             legendgroup="thr", showlegend=(r_i == 1), line=dict(color=D["colors"]["thr_drop"], width=2.5, dash="dot"),
                             marker=dict(size=11, symbol="diamond", color=D["colors"]["thr_drop"]),
                             error_y=dict(type="data", array=[p["sd"] or 0 for p in thr], thickness=1.2, width=4),
                             text=[p["label"] for p in thr], hovertemplate="%{text}<br>delivery %{x:.3f}<br>NLL %{y:.4f}<extra></extra>"))
    for ref, sym in (("rne", "square"), ("kahan", "x"), ("fp32_master", "triangle-up")):
        pts = [p for p in P if p["series"] == ref]
        if pts:
            p = pts[0]
            fig.add_trace(go.Scatter(x=[p["x"]], y=[p["y"]], mode="markers", name=p["label"], legendgroup=ref, showlegend=(r_i == 1),
                                     marker=dict(size=12, symbol=sym, color=D["colors"][ref], line=dict(width=1.5)),
                                     error_y=dict(type="data", array=[p["sd"] or 0], thickness=1.2, width=4),
                                     hovertemplate=p["label"] + "<br>delivery %{x:.3f}<br>NLL %{y:.4f}<extra></extra>"))
    add_reference_line(fig, y=D["bar"], label="fp32 master at lr 5e-5")
    fig.update_yaxes(title="Held-out NLL (nats/token)")
    fig.update_xaxes(title="Cumulative delivery ‖R‖/‖I‖", range=[0.1, 1.05])
apply_theme(fig, height=440)
"""


def fig_dvl(curves, verdicts, colors, root, lrk):
    lrs = [lrk]
    points = {}
    for k in lrs:
        pts = []
        for c, e in curves[k].items():
            if e.get("delivery_mean") is None or e.get("nll_mean") is None:
                continue
            if c.startswith("keno_drop") and c != "keno_drop_range":
                series, label = "keno_drop", f"keno drop {c.split('_')[-1]}%" + ("" if e["n_seeds"] == 5 else f" ({e['n_seeds']} seeds)")
            elif c.startswith("thr_drop"):
                series, label = "thr_drop", f"threshold drop matched to {c.split('_')[-1]}% (θ={e.get('theta'):.3g})" if e.get("theta") else f"threshold drop {c.split('_')[-1]}"
            elif c in ("rne", "kahan", "fp32_master"):
                series, label = c, LABEL[c]
            else:
                continue
            pts.append({"series": series, "label": label, "x": e["delivery_mean"], "y": e["nll_mean"], "n": e["n_seeds"], "sd": e.get("nll_sd")})
        points[k] = pts
    data = {"lrs": lrs, "lr_label": LR_LABEL, "points": points, "colors": colors,
            "bar": verdicts["q2_tuned_keno_vs_tuned_exact"]["bar"]["mean"],
            "alt": f"Held-out NLL against cumulative delivery at lr {LR_LABEL[lrk]}: the random-pick keno-drop curve at "
                   "15/25/50/75%, the threshold-drop points calibrated to the same delivery, plus plain rounding, kahan and fp32 master; "
                   "error bars are seed standard deviations and the horizontal line is fp32 master at lr 5e-5."}
    return render(f"delivery_vs_loss_lr{LR_LABEL[lrk]}", BODY_DVL, data, root)


# --------------------------------------------------------------------------- 3. Q2 vs master
BODY_Q2 = """
rows = D["rows"]
fig = go.Figure()
for series, sym in (("keno_drop", "circle"), ("thr_drop", "diamond")):
    sub = [r for r in rows if r["series"] == series]
    fig.add_trace(go.Scatter(x=[r["label"] for r in sub], y=[r["mean"] for r in sub], mode="markers", name=D["names"][series],
                             marker=dict(color=D["colors"][series], symbol=sym, size=[14 if r["primary"] else 10 for r in sub],
                                         line=dict(width=[2.5 if r["primary"] else 1 for r in sub], color=D["colors"][series])),
                             error_y=dict(type="data", symmetric=False, array=[r["hi"] - r["mean"] for r in sub],
                                          arrayminus=[r["mean"] - r["lo"] for r in sub], thickness=2, width=7, color=D["colors"][series]),
                             text=[f"{r['label']}: {r['verdict']}" + (" (primary)" if r["primary"] else "") for r in sub],
                             hovertemplate="%{text}<br>mean Δ NLL %{y:.4f}<extra></extra>"))
add_reference_line(fig, y=0, label="fp32 master at lr 5e-5")
add_reference_line(fig, y=0.01, label="+0.01 band")
add_reference_line(fig, y=-0.01, label="−0.01 band")
fig.update_xaxes(title="Configuration", categoryorder="array", categoryarray=[r["label"] for r in rows])
fig.update_yaxes(title="Δ held-out NLL vs fp32 master @ 5e-5 (nats)")
apply_theme(fig, height=460)
"""


def fig_q2(verdicts, colors, root):
    q2 = verdicts["q2_tuned_keno_vs_tuned_exact"]
    rows = []
    items = list(q2["exploratory"].items())
    if q2["primary"]:
        items.append((f"{q2['primary']['condition']}@{q2['primary']['lr']}", q2["primary"]))
    for key, iv in items:
        cond, lrk = key.split("@")
        if iv["lo"] is None:
            continue
        series = "keno_drop" if cond.startswith("keno") else "thr_drop"
        rows.append({"key": key, "label": f"{'keno' if series == 'keno_drop' else 'thr'} {cond.split('_')[-1]}% @ {LR_LABEL[lrk]}",
                     "series": series, "mean": iv["mean"], "lo": iv["lo"], "hi": iv["hi"], "verdict": iv["verdict"],
                     "primary": (cond, lrk) == (q2["primary"]["condition"], q2["primary"]["lr"]) if q2["primary"] else False})
    rows.sort(key=lambda r: (r["series"], -float(r["key"].split("@")[1]), int(r["key"].split("@")[0].split("_")[-1])))
    data = {"rows": rows, "colors": colors, "names": {"keno_drop": "random pick (keno drop)", "thr_drop": "magnitude pick (threshold drop)"},
            "alt": "Dot-and-interval chart of each five-seed configuration's paired held-out NLL difference against fp32 master weights at "
                   "lr 5e-5, with 90% Student-t intervals; the larger marker is the pre-registered keno drop 50% at lr 1e-4."}
    return render("q2_vs_master", BODY_Q2, data, root)


# --------------------------------------------------------------------------- 4. never moved by family
BODY_FAM = """
fig = go.Figure()
for g in D["groups"]:
    fig.add_trace(go.Bar(x=[D["fam_label"][f] for f in D["fams"]], y=g["values"], name=g["label"],
                         marker=dict(color=g["color"], opacity=g["opacity"], pattern=dict(shape=g["pattern"])),
                         hovertemplate=g["label"] + "<br>%{x}: %{y:.3f}<extra></extra>"))
fig.update_layout(barmode="group")
fig.update_xaxes(title="Parameter family")
fig.update_yaxes(title="Never-moved coordinate fraction", range=[0, 1])
apply_theme(fig, height=420)
"""


def fig_family(curves, colors, root, lrk, lv):
    groups = []
    for series, pat in (("keno_drop", ""), ("thr_drop", "/")):
        e = curves[lrk].get(f"{series}_{lv}")
        if not e or "families" not in e:
            continue
        groups.append({"label": f"{'random' if series == 'keno_drop' else 'magnitude'} pick, {lv}% level", "color": colors[series],
                       "opacity": 1.0, "pattern": pat, "values": [e["families"][f]["never_moved"] for f in FAMS]})
    data = {"fams": FAMS, "fam_label": FAM_LABEL, "groups": groups, "lr": lrk, "level": lv,
            "alt": f"Grouped bars of the fraction of coordinates never moved over 3,000 steps per parameter family (attention, MLP, embedding, "
                   f"norm) at lr {LR_LABEL[lrk]}, random pick against the magnitude pick matched to the {lv}% level; seed means."}
    return render(f"never_moved_family_{lv}_lr{LR_LABEL[lrk]}", BODY_FAM, data, root)


# --------------------------------------------------------------------------- 5. delivery over steps
BODY_STEPS = """
fig = go.Figure()
for s in D["series"]:
    fig.add_trace(go.Scatter(x=s["steps"], y=s["y"], mode="lines", name=s["label"],
                             line=dict(color=s["color"], width=2.2 if s["kind"] == "keno_drop" else 2.2, dash=s["dash"]),
                             hovertemplate=s["label"] + "<br>step %{x}<br>delivery %{y:.3f}<extra></extra>"))
fig.update_xaxes(title="Training step")
fig.update_yaxes(title="Cumulative delivery ‖R‖/‖I‖")
apply_theme(fig, height=440)
"""


def fig_steps(series, colors, root, lrk, lv):
    out = []
    for kind, dash in (("keno_drop", "solid"), ("thr_drop", "dot")):
        e = series[lrk].get(f"{kind}_{lv}")
        if not e:
            continue
        out.append({"label": f"{'random' if kind == 'keno_drop' else 'magnitude'} pick, {lv}% level ({e['n_seeds']} seeds)", "kind": kind,
                    "color": colors[kind], "dash": dash, "steps": e["steps"], "y": e["cumulative_length_ratio"], "n_seeds": e["n_seeds"]})
    data = {"series": out, "lr": lrk, "level": lv,
            "alt": f"Lines of cumulative delivery against training step at lr {LR_LABEL[lrk]} for the random pick and the calibrated "
                   f"magnitude pick at the {lv}% level; seed means over this run's records."}
    return render(f"delivery_over_steps_{lv}_lr{LR_LABEL[lrk]}", BODY_STEPS, data, root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--root", default="figures")
    args = ap.parse_args()
    verdicts = json.load(open(os.path.join(args.analysis, "verdicts.json")))
    curves = json.load(open(os.path.join(args.analysis, "curves.json")))
    series = json.load(open(os.path.join(args.analysis, "step_series.json")))
    colors = entity_colors(args.root)
    fig_q1(verdicts, colors, args.root)
    for lrk in ("0.0002", "0.0001"):
        fig_dvl(curves, verdicts, colors, args.root, lrk)
    fig_q2(verdicts, colors, args.root)
    for lrk, lv in (("0.0002", "25"), ("0.0001", "50")):
        fig_family(curves, colors, args.root, lrk, lv)
        fig_steps(series, colors, args.root, lrk, lv)
    print("figures written under", args.root)


if __name__ == "__main__":
    main()
