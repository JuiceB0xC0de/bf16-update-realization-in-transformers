"""Render the report figures from results/analysis (pod-side, silico-figures).

Usage: uv run --no-sync python src/prepare_figures.py --analysis results/analysis [--root figures]
Writes figures/<name>/{<name>.html, data.json, alt.txt, plot.py}:
  paired_vs_rne        paired held-out NLL difference keno writer - plain rounding (rne), per lr, seed mean + per-seed points
  delivery_vs_loss     held-out NLL against cumulative delivery (length ratio), keno writers and previous-run reference points
  tab_vs_master_drop   Q2: keno_tab - fp32_master and keno_tab - keno_drop per fraction and lr
  delivery_over_steps_{drop,tab}  cumulative length ratio against training step at lr 2e-4, one figure per mode
  never_moved_family   never-moved coordinate fraction per parameter family at lr 2e-4
Each bundle's plot.py is the exact code used here (the body string is exec'd to build and written to disk).
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from silico_figures import EDITORIAL_8, add_reference_line, apply_theme, save_figure_bundle

KENO = ["keno_drop_25", "keno_drop_50", "keno_drop_75", "keno_drop_range",
        "keno_tab_25", "keno_tab_50", "keno_tab_75", "keno_tab_range"]
LABEL = {"keno_drop_25": "drop 25%", "keno_drop_50": "drop 50%", "keno_drop_75": "drop 75%", "keno_drop_range": "drop 30-70%",
         "keno_tab_25": "tab 25%", "keno_tab_50": "tab 50%", "keno_tab_75": "tab 75%", "keno_tab_range": "tab 30-70%",
         "rne": "plain rounding (rne)", "fp32_master": "fp32 master", "kahan": "kahan", "torch_adamw": "torch AdamW"}
FAMS = ["attention", "mlp", "embedding", "norm"]
FAM_LABEL = {"attention": "attention", "mlp": "MLP", "embedding": "embedding", "norm": "norm"}
LR_LABEL = {"0.0002": "2e-4", "0.0001": "1e-4", "5e-05": "5e-5"}


def entity_colors(root):
    """Continue the previous experiment's mapping (rne, fp32_master, kahan keep their colors); keno writers are
    coloured by mode (drop / tab); the draw fraction is carried by position, text or dash, never by colour."""
    path = os.path.join(root, "entity_colors.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    colors = {"kahan": EDITORIAL_8[0], "rne": EDITORIAL_8[1], "fp32_master": EDITORIAL_8[2],
              "keno_drop": EDITORIAL_8[4], "keno_tab": EDITORIAL_8[7], "torch_adamw": EDITORIAL_8[5]}
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as f:
        json.dump(colors, f, indent=1)
    return colors


def color_of(c, colors):
    if c in colors:
        return colors[c]
    return colors["keno_drop"] if c.startswith("keno_drop") else colors["keno_tab"]


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


# --------------------------------------------------------------------------- 1. paired vs rne
BODY_PAIRED = """
lrs = D["lrs"]
fig = make_subplots(rows=len(lrs), cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    subplot_titles=[f"Learning rate {D['lr_label'][k]}" for k in lrs])
for r_i, k in enumerate(lrs, start=1):
    for g in D["groups"]:
        pts = [r for r in D["rows"] if r["lr"] == k and r["group"] == g["name"]]
        xs, ys, sym, size, txt = [], [], [], [], []
        for r in pts:
            focal = r["condition"] == D["focal"]
            xs.append(r["label"]); ys.append(r["mean"]); sym.append(g["symbol"]); size.append(15 if focal else 11)
            txt.append(f"{r['label']}: two-seed mean")
            for sd, v in zip(r["seeds"], r["per_seed"]):
                xs.append(r["label"]); ys.append(v); sym.append(g["symbol"] + "-open"); size.append(7)
                txt.append(f"{r['label']}: seed {sd}")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=g["label"], legendgroup=g["name"], showlegend=(r_i == 1),
                                 marker=dict(color=g["color"], symbol=sym, size=size, line=dict(width=1.5)),
                                 text=txt, hovertemplate="%{text}<br>Δ NLL %{y:.4f}<extra></extra>"), row=r_i, col=1)
    add_reference_line(fig, y=0, label="same loss as plain rounding" if r_i == 1 else "", row=r_i, col=1)
    add_reference_line(fig, y=0.02, label="+0.02" if r_i == 1 else "", row=r_i, col=1)
    add_reference_line(fig, y=-0.02, label="−0.02" if r_i == 1 else "", row=r_i, col=1)
    fig.update_yaxes(title="Δ held-out NLL vs rne (nats)", row=r_i, col=1)
fig.update_xaxes(title="Writer", row=len(lrs), col=1)
apply_theme(fig, height=300 * len(lrs) + 60)
"""


def keno_groups(colors):
    return [{"name": "keno_drop", "label": "keno drop (undelivered request discarded)", "color": colors["keno_drop"], "symbol": "circle"},
            {"name": "keno_tab", "label": "keno tab (undelivered request banked)", "color": colors["keno_tab"], "symbol": "square"}]


def fig_paired(pairs, colors, root):
    lrs = [k for k in sorted(pairs.keys(), key=float, reverse=True) if any(x.startswith("keno_") for x in pairs[k])]
    rows = []
    for k in lrs:
        for c in KENO:
            d = pairs[k].get(f"{c}-rne")
            if d:
                rows.append({"lr": k, "condition": c, "label": LABEL[c], "group": "keno_drop" if "drop" in c else "keno_tab",
                             "mean": d["mean"], "per_seed": d["per_seed"], "seeds": d["seeds"]})
    data = {"lrs": lrs, "lr_label": LR_LABEL, "rows": rows, "focal": "keno_drop_25", "groups": keno_groups(colors),
            "alt": "Dot chart of the paired held-out NLL difference between each keno writer and plain bfloat16 rounding "
                   "at two learning rates: filled marker is the two-seed mean, open markers are the per-seed values, "
                   "with reference lines at 0 and ±0.02."}
    return render("paired_vs_rne", BODY_PAIRED, data, root)


# --------------------------------------------------------------------------- 2. delivery vs loss
BODY_DVL = """
lrs = D["lrs"]
fig = make_subplots(rows=len(lrs), cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    subplot_titles=[f"Learning rate {D['lr_label'][k]}" for k in lrs])
for r_i, k in enumerate(lrs, start=1):
    for g in D["groups"]:
        pts = [p for p in D["points"] if p["lr"] == k and p["group"] == g["name"]]
        if not pts:
            continue
        xs, ys, sym, size, txt, lab = [], [], [], [], [], []
        for p in pts:
            focal = p["condition"] == D["focal"]
            xs.append(p["x"]); ys.append(p["y"]); sym.append(g["symbol"]); size.append(15 if focal else 11)
            txt.append(f"{p['label']}: two-seed mean"); lab.append(p["short"])
            for xv, yv in zip(p["x_per_seed"], p["y_per_seed"]):
                xs.append(xv); ys.append(yv); sym.append(g["symbol"] + "-open"); size.append(6); txt.append(p["label"] + ": one seed"); lab.append("")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=lab, textposition=g["textpos"],
                                 marker=dict(color=g["color"], symbol=sym, size=size, line=dict(width=1.2)),
                                 name=g["label"], legendgroup=g["name"], showlegend=(r_i == 1),
                                 customdata=txt, hovertemplate="%{customdata}<br>delivery %{x:.3f}, NLL %{y:.4f}<extra></extra>"),
                      row=r_i, col=1)
    add_reference_line(fig, y=D["bar"], label=f"{D['bar']:.3f}: fp32 master at lr 5e-5" if r_i == 1 else "", row=r_i, col=1)
    fig.update_yaxes(title="Held-out NLL (nats per token)", row=r_i, col=1)
fig.update_xaxes(title="Cumulative delivery ‖R‖/‖I‖ (fraction of the request written)", range=[0, 1.08], row=len(lrs), col=1)
apply_theme(fig, height=340 * len(lrs) + 60)
"""


def fig_dvl(dvl, verdicts, colors, root):
    lrs = [k for k in sorted(dvl.keys(), key=float, reverse=True) if k != "5e-05"]
    groups = [dict(g, textpos="top center" if g["name"] == "keno_drop" else "bottom center") for g in keno_groups(colors)]
    groups += [{"name": "rne", "label": "plain rounding (previous run)", "color": colors["rne"], "symbol": "diamond", "textpos": "bottom center"},
               {"name": "fp32_master", "label": "fp32 master (previous run)", "color": colors["fp32_master"], "symbol": "diamond", "textpos": "top center"},
               {"name": "kahan", "label": "kahan (previous run)", "color": colors["kahan"], "symbol": "diamond", "textpos": "bottom center"}]
    short = {"25": "25%", "50": "50%", "75": "75%", "range": "30-70%"}
    points = []
    for k in lrs:
        for c, e in dvl[k].items():
            if e.get("nll_mean") is None or c == "torch_adamw":
                continue
            grp = "keno_drop" if c.startswith("keno_drop") else "keno_tab" if c.startswith("keno_tab") else c
            seeds = sorted(e["nll_per_seed"].keys(), key=int)
            xs = [e["cum_length_ratio_per_seed"][s] for s in seeds] if "cum_length_ratio_per_seed" in e else []
            ys = [e["nll_per_seed"][s] for s in seeds] if xs else []
            points.append({"lr": k, "condition": c, "group": grp, "label": LABEL[c], "x": e["cum_length_ratio"], "y": e["nll_mean"],
                           "short": short[c.split("_")[2]] if c.startswith("keno") else LABEL[c].split(" (")[0],
                           "x_per_seed": xs, "y_per_seed": ys, "projection": e.get("cum_projection"),
                           "never_moved": e.get("never_moved")})
    data = {"lrs": lrs, "lr_label": LR_LABEL, "groups": groups, "points": points, "bar": verdicts["q3_bar"], "focal": "keno_drop_25",
            "alt": "Scatter of held-out NLL against cumulative delivery for the eight keno writers and the previous run's "
                   "plain rounding, kahan and fp32 master points at learning rates 2e-4 and 1e-4, with the 1.150 line."}
    return render("delivery_vs_loss", BODY_DVL, data, root)


# --------------------------------------------------------------------------- 3. Q2 tab vs master / drop
BODY_Q2 = """
lrs = D["lrs"]
fig = make_subplots(rows=len(lrs), cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    subplot_titles=[f"Learning rate {D['lr_label'][k]}" for k in lrs])
for r_i, k in enumerate(lrs, start=1):
    for s in D["series"]:
        rows = [r for r in D["rows"] if r["lr"] == k and r["series"] == s["name"]]
        xs, ys, sym, size, txt = [], [], [], [], []
        for r in rows:
            xs.append(r["x"]); ys.append(r["mean"]); sym.append(s["symbol"]); size.append(11); txt.append(f"{s['label']} at {r['x']}: two-seed mean")
            for v in r["per_seed"]:
                xs.append(r["x"]); ys.append(v); sym.append(s["symbol"] + "-open"); size.append(7); txt.append(f"{s['label']} at {r['x']}: one seed")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=s["color"], symbol=sym, size=size, line=dict(width=1.5)),
                                 name=s["label"], legendgroup=s["name"], showlegend=(r_i == 1),
                                 text=txt, hovertemplate="%{text}<br>Δ NLL %{y:.4f}<extra></extra>"), row=r_i, col=1)
    add_reference_line(fig, y=0, label="no difference" if r_i == 1 else "", row=r_i, col=1)
    add_reference_line(fig, y=0.01, label="+0.01" if r_i == 1 else "", row=r_i, col=1)
    add_reference_line(fig, y=-0.01, label="−0.01" if r_i == 1 else "", row=r_i, col=1)
    fig.update_yaxes(title="Δ held-out NLL (nats)", row=r_i, col=1)
fig.update_xaxes(title="Draw fraction", row=len(lrs), col=1)
apply_theme(fig, height=300 * len(lrs) + 60)
"""


def fig_q2(pairs, colors, root):
    lrs = [k for k in sorted(pairs.keys(), key=float, reverse=True) if any(x.startswith("keno_") for x in pairs[k])]
    xs = {"25": "25%", "50": "50%", "75": "75%", "range": "30-70%"}
    rows = []
    for k in lrs:
        for fr in ("25", "50", "75", "range"):
            a = pairs[k].get(f"keno_tab_{fr}-fp32_master")
            b = pairs[k].get(f"keno_tab_{fr}-keno_drop_{fr}")
            if a:
                rows.append({"lr": k, "series": "tab_minus_master", "x": xs[fr], "mean": a["mean"], "per_seed": a["per_seed"]})
            if b:
                rows.append({"lr": k, "series": "tab_minus_drop", "x": xs[fr], "mean": b["mean"], "per_seed": b["per_seed"]})
    series = [{"name": "tab_minus_master", "label": "tab − fp32 master", "color": colors["fp32_master"], "symbol": "circle"},
              {"name": "tab_minus_drop", "label": "tab − drop (same fraction)", "color": colors["keno_drop"], "symbol": "square"}]
    data = {"lrs": lrs, "lr_label": LR_LABEL, "rows": rows, "series": series,
            "alt": "Dot chart of the paired held-out NLL difference of each keno tab writer from fp32 master weights and from the "
                   "keno drop writer at the same draw fraction, at two learning rates, with reference lines at 0 and ±0.01."}
    return render("tab_vs_master_drop", BODY_Q2, data, root)


# --------------------------------------------------------------------------- 4. delivery over steps
BODY_STEPS = """
fig = go.Figure()
for s in D["series"]:
    fig.add_trace(go.Scatter(x=s["steps"], y=s["y"], mode="lines", name=s["label"],
                             line=dict(color=s["color"], width=s["width"], dash=s["dash"]),
                             hovertemplate=s["label"] + " step %{x}: %{y:.3f}<extra></extra>"))
fig.update_yaxes(title="Cumulative delivery ‖R‖/‖I‖", range=[0, 1.05])
fig.update_xaxes(title="Training step")
apply_theme(fig, height=420)
"""


def fig_steps(series, colors, root, lrk="0.0002"):
    """One figure per mode (drop, tab). Within a figure the draw fraction is an ordered variable, so 25/50/75% take
    three steps of the sequential ramp (light to dark) and the unordered 30-70% writer takes the mode colour, dashed;
    plain rounding keeps its entity colour, dotted."""
    import plotly.colors as pc
    from silico_figures import SEQUENTIAL
    ramp = dict(zip(("25", "50", "75"), pc.sample_colorscale(SEQUENTIAL, [0.45, 0.72, 0.92])))
    S = series[lrk]
    short = {"25": "25%", "50": "50%", "75": "75%", "range": "30-70%"}
    figs = {}
    for mode in ("drop", "tab"):
        out = []
        for c in KENO:
            if c not in S or c.split("_")[1] != mode:
                continue
            fr = c.split("_")[2]
            out.append({"name": c, "label": LABEL[c], "short": short[fr], "mode": mode, "steps": S[c]["steps"],
                        "y": S[c]["cumulative_length_ratio"], "color": ramp.get(fr, color_of(c, colors)),
                        "width": 2.6 if c == "keno_drop_25" else 1.8, "dash": "dash" if fr == "range" else "solid"})
        if "rne" in S:
            out.append({"name": "rne", "label": LABEL["rne"], "short": "rne", "mode": "ref", "steps": S["rne"]["steps"],
                        "y": S["rne"]["cumulative_length_ratio"], "color": colors["rne"], "width": 1.8, "dash": "dot"})
        data = {"lr": LR_LABEL[lrk], "mode": mode, "series": out,
                "alt": f"Line chart of cumulative delivery (length ratio) against training step for the keno {mode} writers at "
                       f"learning rate {LR_LABEL[lrk]}, with plain rounding dotted; the 25, 50 and 75% writers run light to dark "
                       "on the sequential ramp and the 30-70% writer is dashed."}
        figs[mode] = render(f"delivery_over_steps_{mode}", BODY_STEPS, data, root)
    return figs


# --------------------------------------------------------------------------- 5. never-moved by family (heatmap)
BODY_FAM = """
fig = go.Figure(go.Heatmap(z=D["z"], x=D["families"], y=D["writers"], zmin=0, zmax=1, colorscale=D["colorscale"],
                           colorbar=dict(title="Never-moved fraction"), text=[[f"{v:.2f}" for v in row] for row in D["z"]],
                           texttemplate="%{text}", hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>"))
fig.update_xaxes(title="Parameter family")
fig.update_yaxes(title="Writer", autorange="reversed")
apply_theme(fig, height=460)
"""


def fig_family(dvl, colors, root, lrk="0.0002"):
    from silico_figures import SEQUENTIAL
    E = dvl[lrk]
    order = ["rne", "keno_drop_25", "keno_drop_50", "keno_drop_75", "keno_drop_range", "keno_tab_25", "keno_tab_50",
             "keno_tab_75", "keno_tab_range", "fp32_master"]
    writers, z = [], []
    for c in order:
        if c not in E or "families" not in E[c]:
            continue
        writers.append(LABEL[c])
        z.append([E[c]["families"][f]["never_moved"] for f in FAMS])
    data = {"lr": LR_LABEL[lrk], "families": [FAM_LABEL[f] for f in FAMS], "writers": writers, "z": z, "colorscale": SEQUENTIAL,
            "alt": f"Heatmap of the never-moved coordinate fraction per parameter family at learning rate {LR_LABEL[lrk]} "
                   "for plain rounding, the eight keno writers and fp32 master weights (sequential scale from 0 to 1)."}
    return render("never_moved_family", BODY_FAM, data, root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--root", default="figures")
    args = ap.parse_args()
    A = {}
    for n in ("verdicts", "paired_differences", "delivery_vs_loss", "step_series"):
        with open(os.path.join(args.analysis, n + ".json")) as f:
            A[n] = json.load(f)
    colors = entity_colors(args.root)
    fig_paired(A["paired_differences"], colors, args.root)
    fig_dvl(A["delivery_vs_loss"], A["verdicts"], colors, args.root)
    fig_q2(A["paired_differences"], colors, args.root)
    fig_steps(A["step_series"], colors, args.root)
    fig_family(A["delivery_vs_loss"], colors, args.root)
    print("figures written under", args.root)


if __name__ == "__main__":
    main()
