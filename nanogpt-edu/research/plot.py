"""Plot autoresearch progress from ledger.tsv → progress.png.

    python plot.py            # writes progress.png

Two panels (improving on autokernel's redundant 2-panel single-metric chart):

  * Top    — running-best **val_bpb** (quality, lower better): a smooth
             monotone PCHIP curve through real improvements + a 3-state scatter
             (kept / discarded / crash) + a gain band, with milestone labels.
  * Bottom — **quality-vs-cost Pareto**: val_bpb (y) against throughput (x),
             every experiment a point, kept ones highlighted, the Pareto
             frontier traced. This is the tradeoff frontier a single-metric
             chart structurally can't show — the visual form of this repo's
             "iso-param ≠ iso-activation / sizing-fact" ethos.

PCHIP (monotone cubic) is used for the smooth-yet-honest running-best line; it
passes exactly through every real improvement and never overshoots between them.
Uses scipy if present, else a tiny vendored numpy implementation (keeps the core
deps lean — scipy is not in requirements.txt).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

try:  # PCHIP: scipy if available, else the vendored fallback below.
    from scipy.interpolate import PchipInterpolator as _SciPCHIP
except Exception:  # noqa: BLE001
    _SciPCHIP = None

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.tsv"
OUTPUT = HERE / "progress.png"

COLORS = {"keep": "#22c55e", "discard": "#cbd5e1", "crash": "#ef4444"}
BEST_LINE = "#15803d"
MILESTONE = "#0f766e"
PARETO = "#7c3aed"


def _pchip(x: np.ndarray, y: np.ndarray, samples: int = 400):
    """Monotone cubic (PCHIP) interpolation. scipy if present, else vendored.

    The vendored path implements the Fritsch–Carlson monotone-cubic Hermite
    scheme: secant slopes, harmonic-mean tangents, zeroed at local extrema — so
    the curve is smooth and never overshoots between data points.
    """
    if len(x) < 2:
        return x, y
    xd = np.linspace(x[0], x[-1], samples)
    if _SciPCHIP is not None:
        return xd, _SciPCHIP(x, y)(xd)
    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros_like(y)
    m[1:-1] = np.where(delta[:-1] * delta[1:] > 0,
                       2 / (1 / delta[:-1] + 1 / delta[1:]), 0.0)
    m[0], m[-1] = delta[0], delta[-1]
    yd = np.empty_like(xd)
    idx = np.clip(np.searchsorted(x, xd) - 1, 0, len(x) - 2)
    for k in range(len(xd)):
        i = idx[k]
        t = (xd[k] - x[i]) / h[i]
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        yd[k] = (h00 * y[i] + h10 * h[i] * m[i]
                 + h01 * y[i + 1] + h11 * h[i] * m[i + 1])
    return xd, yd


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"no {path.name}; run loop.py first to populate the ledger")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for i, r in enumerate(rows, 1):
        r["experiment"] = int(r.get("experiment") or i)
        for k in ("val_bpb", "tok_per_s", "vram_mb", "params_m", "gen_gap"):
            try:
                r[k] = float(r[k])
            except (ValueError, KeyError, TypeError):
                r[k] = math.nan
        r["status"] = (r.get("status") or "").strip().lower()
    return rows


def pareto_front(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Frontier minimising both x (cost) and y (val_bpb): no other point is
    better-or-equal on both. Returns the frontier sorted by x."""
    front = []
    for p in sorted(pts):
        if not any(q[0] <= p[0] and q[1] < p[1] for q in pts):
            front.append(p)
    # keep only the lower envelope (monotone non-increasing y as x grows)
    out, best_y = [], math.inf
    for x, y in front:
        if y < best_y - 1e-12:
            out.append((x, y)); best_y = y
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    rows = load(LEDGER)
    for r in rows:
        r["valid"] = math.isfinite(r["val_bpb"]) and r["status"] != "crash"

    # running-best val_bpb over kept experiments
    best = math.inf
    run_best = []
    for r in rows:
        if r["status"] == "keep" and r["valid"]:
            best = min(best, r["val_bpb"])
        run_best.append(best if math.isfinite(best) else math.nan)
    for r, b in zip(rows, run_best):
        r["run_best"] = b

    kept = [r for r in rows if r["status"] == "keep" and r["valid"]]
    if not kept:
        raise SystemExit("no kept experiments yet — nothing to plot")
    baseline = kept[0]["val_bpb"]
    best_bpb = min(r["val_bpb"] for r in kept)

    # milestone change-points (where the running best improved)
    miles, prev = [], math.inf
    for r in rows:
        if r["status"] == "keep" and r["valid"] and r["val_bpb"] < prev - 1e-9:
            miles.append(r); prev = r["val_bpb"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.patch.set_facecolor("#fafafa")

    # ---- Panel 1: running-best val_bpb (quality) ----
    ax = axes[0]
    ax.set_facecolor("#ffffff")
    xs = np.array([r["experiment"] for r in rows if math.isfinite(r["run_best"])], float)
    ys = np.array([r["run_best"] for r in rows if math.isfinite(r["run_best"])], float)
    # dedup consecutive equal y for a clean monotone fit
    keep_pts = [0] + [i for i in range(1, len(ys)) if abs(ys[i] - ys[i - 1]) > 1e-12]
    xp, yp = xs[keep_pts], ys[keep_pts]
    if xp[-1] != xs[-1]:
        xp, yp = np.append(xp, xs[-1]), np.append(yp, yp[-1])
    xd, yd = _pchip(xp, yp)
    ax.fill_between(xd, baseline, yd, color=BEST_LINE, alpha=0.16, label="improvement vs baseline")
    ax.plot(xd, yd, color=BEST_LINE, lw=2.8, zorder=4, label="running best val_bpb")
    for status in ("discard", "crash", "keep"):
        part = [r for r in rows if r["status"] == status and (r["valid"] or status == "crash")]
        if not part:
            continue
        ax.scatter([r["experiment"] for r in part],
                   [r["val_bpb"] if math.isfinite(r["val_bpb"]) else baseline for r in part],
                   c=COLORS[status], s=70 if status == "keep" else 30,
                   edgecolors="#14532d" if status == "keep" else "none",
                   linewidths=0.8, alpha=0.95 if status == "keep" else 0.7,
                   zorder=3 if status == "keep" else 2)
    for m in (miles[:1] + miles[-1:] if len(miles) > 1 else miles):
        ax.annotate(f"{m['val_bpb']:.3f}", (m["experiment"], m["val_bpb"]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=9, color=MILESTONE, fontweight="bold")
    ax.axhline(baseline, color="#94a3b8", ls="--", lw=1, alpha=0.8, label="baseline")
    ax.set_ylabel("val_bpb (bits/byte, lower better)")
    gain = (baseline - best_bpb) / baseline * 100 if baseline else 0.0
    ax.set_title(f"nanogpt-edu autoresearch — val_bpb {baseline:.4f} → {best_bpb:.4f} "
                 f"({gain:.1f}% better)  ·  {len(kept)} kept / {len(rows)} runs",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("experiment #")
    ax.legend(loc="upper right", framealpha=0.95)

    # ---- Panel 2: quality vs cost Pareto ----
    ax = axes[1]
    ax.set_facecolor("#ffffff")
    valid = [r for r in rows if r["valid"] and math.isfinite(r["tok_per_s"]) and r["tok_per_s"] > 0]
    for status in ("discard", "keep"):
        part = [r for r in valid if r["status"] == status]
        if not part:
            continue
        ax.scatter([r["tok_per_s"] / 1e3 for r in part], [r["val_bpb"] for r in part],
                   c=COLORS[status], s=80 if status == "keep" else 34,
                   edgecolors="#14532d" if status == "keep" else "none",
                   linewidths=0.8, alpha=0.95 if status == "keep" else 0.65,
                   label=("kept" if status == "keep" else "discarded"), zorder=3)
    pts = [(r["tok_per_s"] / 1e3, r["val_bpb"]) for r in valid]
    front = pareto_front(pts)
    if len(front) >= 2:
        fx, fy = zip(*front)
        ax.plot(fx, fy, color=PARETO, lw=2.2, marker="o", ms=5, zorder=4,
                label="Pareto frontier (quality vs speed)")
    ax.set_xlabel("throughput (k tok/s, higher better →)")
    ax.set_ylabel("val_bpb (lower better)")
    ax.set_title("quality–cost tradeoff frontier", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.legend(loc="upper right", framealpha=0.95)

    summary = (f"{len(rows)} experiments  ·  {len(kept)} kept  ·  "
               f"{sum(1 for r in rows if r['status']=='crash')} crashed  ·  "
               f"best val_bpb {best_bpb:.4f} ({gain:.1f}% vs baseline)")
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10, color="#334155")
    handles = [mpatches.Patch(color=COLORS["keep"], label="kept (improved)"),
               mpatches.Patch(color=COLORS["discard"], label="discarded"),
               mpatches.Patch(color=COLORS["crash"], label="crash / gate-fail")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.998),
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    fig.savefig(OUTPUT, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
