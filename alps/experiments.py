"""Experiments for the rotation-corrected ALPS script.

Subcommands:

- ``direct``: run ``alps.py`` on a dataset.
- ``tilt``: emulate a head tilt (rotate the anatomy and the b-vectors about the
  image centre) and rerun the full pipeline at each angle.
- ``regdiag``: register a tilted mean b0 to the template and score the rotation
  and the ROI placement against the known tilt.
- ``sensitivity``: perturb each soft-score centre and width of the ``data``
  method and rerun the ROI step only (fit and registration done once).
- ``summary``: collect all results into CSV tables and a Markdown report.

Results go to ``alps_results/<dataset>/<experiment>/``. Usage::

    python alps/experiments.py direct --dataset hbn
    python alps/experiments.py tilt --dataset hbn --angles 0 0.5 5 10 -5 -10
    python alps/experiments.py regdiag --dataset hbn
    python alps/experiments.py sensitivity --dataset hbn
    python alps/experiments.py summary
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time

from datasets import DATASETS, dataset_paths
import numpy as np
from scipy.spatial.transform import Rotation

import alps as A

RESULTS = Path("alps_results")  # set from --results-dir in main()

# Registration variants for ``regdiag``. Both images are skull-stripped.
REG_VARIANTS = {
    "rigid": ("center_of_mass", "translation", "rigid"),
    "affine": ("center_of_mass", "translation", "rigid", "affine"),
}


# ---------------------------------------------------------------------------
# Tilt emulation
# ---------------------------------------------------------------------------


def tilt_matrix(angle_deg, axis):
    return Rotation.from_euler(axis, angle_deg, degrees=True).as_matrix()


def image_centre(shape, affine):
    """World coordinates of the centre of the voxel grid."""
    return affine[:3, :3] @ ((np.asarray(shape[:3]) - 1) / 2.0) + affine[:3, 3]


def tilt_volume(vol, affine, Rt, *, interpolation="linear"):
    """Rotate a 3D volume by ``Rt`` about its world-space centre."""
    from dipy.align.imaffine import AffineMap

    centre = image_centre(vol.shape, affine)
    # Output at world point x samples the input at Rt^-1 (x - c) + c.
    T = np.eye(4)
    T[:3, :3] = Rt.T
    T[:3, 3] = centre - Rt.T @ centre
    amap = AffineMap(
        T,
        domain_grid_shape=vol.shape[:3],
        domain_grid2world=affine,
        codomain_grid_shape=vol.shape[:3],
        codomain_grid2world=affine,
    )
    return amap.transform(vol, interpolation=interpolation)


def tilt_dwi(data, affine, bvecs, Rt, *, bvec_convention="fsl"):
    """Rotate a 4D DWI and its b-vectors by ``Rt``.

    The b-vectors follow the anatomy, so the tensors in the tissue frame do
    not change.
    """
    tilted = np.stack(
        [tilt_volume(data[..., i], affine, Rt) for i in range(data.shape[-1])],
        axis=-1,
    )
    M = A.bvec_frame_to_world(affine, convention=bvec_convention)
    return tilted, (M.T @ Rt @ M @ bvecs.T).T


def angle_dir(axis, angle):
    return f"{axis}{angle:+.1f}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_dataset(name):
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti

    dwi, bval, bvec, max_bval = dataset_paths(name)
    data, affine = load_nifti(dwi)
    bvals, bvecs = read_bvals_bvecs(bval, bvec)
    if max_bval is not None:
        keep = bvals <= max_bval
        data, bvals, bvecs = data[..., keep], bvals[keep], bvecs[keep]
    return data, affine, bvals, bvecs


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Union of the keys in first-seen order: rows from older runs may lack
    # newer columns.
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_direct(args):
    dwi, bval, bvec, max_bval = dataset_paths(args.dataset)
    argv = [dwi, bval, bvec, "--out-dir", str(RESULTS / args.dataset / "direct")]
    if max_bval is not None:
        argv += ["--max-bval", str(max_bval)]
    A.main(argv=argv)


def cmd_tilt(args):
    from dipy.io.image import save_nifti

    out_dir = RESULTS / args.dataset / "tilt"
    data, affine, bvals, bvecs = load_dataset(args.dataset)
    rows = []
    for angle in args.angles:
        run_dir = out_dir / angle_dir(args.axis, angle)
        run_dir.mkdir(parents=True, exist_ok=True)
        tilted, tbvecs = tilt_dwi(data, affine, bvecs, tilt_matrix(angle, args.axis))
        dwi = run_dir / "dwi.nii.gz"
        save_nifti(dwi, tilted.astype(np.float32), affine)
        np.savetxt(run_dir / "dwi.bval", bvals[None], fmt="%g")
        np.savetxt(run_dir / "dwi.bvec", tbvecs.T, fmt="%.6f")
        out = A.main(
            argv=[
                str(dwi),
                str(run_dir / "dwi.bval"),
                str(run_dir / "dwi.bvec"),
                "--out-dir",
                str(run_dir),
            ]
        )
        if not args.keep_dwi:
            dwi.unlink()
        for (method, frame), res in out["results"].items():
            rows.append(
                {
                    "axis": args.axis,
                    "angle": angle,
                    "tilt_measured": out["frames"]["tilt_deg"],
                    "method": method,
                    "frame": frame,
                    "alps_L": res["L"]["alps"],
                    "alps_R": res["R"]["alps"],
                    "alps_mean": res["mean"],
                    "alps_mean_unweighted": res["mean_unweighted"],
                }
            )
    # Merge with earlier angles so the series can be extended run by run.
    path = out_dir / "tilt.csv"
    if path.exists():
        done = {(r["axis"], float(r["angle"])) for r in rows}
        old = [r for r in read_rows(path) if (r["axis"], float(r["angle"])) not in done]
        rows = old + rows
    rows.sort(key=lambda r: (r["axis"], float(r["angle"])))
    write_rows(rows, path)
    print(f"wrote {path}")


def cmd_regdiag(args):
    from dipy.segment.mask import median_otsu

    data, affine, bvals, bvecs = load_dataset(args.dataset)
    from dipy.core.gradients import gradient_table

    gtab = gradient_table(bvals, bvecs=bvecs)
    b0 = data[..., gtab.b0s_mask].mean(axis=-1)
    del data
    _, mask0 = median_otsu(b0, median_radius=4, numpass=4, dilate=2)
    centre = image_centre(b0.shape, affine)
    tmpl, tmpl_mask, tmpl_aff = A.load_template()
    M = A.bvec_frame_to_world(affine, convention="fsl")

    names = [(h, g) for h in A.ALPS_MNI_COORDS for g in ("proj", "assoc")]
    mni = np.array([A.ALPS_MNI_COORDS[h][g] for h, g in names], float)
    mni_h = np.c_[mni, np.ones(len(mni))]
    angles = args.angles if 0.0 in args.angles else [0.0, *args.angles]

    tilted = {}
    for angle in angles:
        Rt = tilt_matrix(angle, args.axis)
        tb0 = tilt_volume(b0, affine, Rt)
        geom = tilt_volume(mask0.astype(float), affine, Rt, interpolation="nearest")
        _, tmask = median_otsu(tb0, median_radius=4, numpass=4, dilate=2)
        lost = 1.0 - (geom > 0.5).sum() / mask0.sum()
        tilted[angle] = (Rt, tb0, tmask, lost)
        print(f"angle {angle:+5.1f}: brain mask lost to FOV {100 * lost:5.2f}%")

    rows = []
    for variant in args.variants:
        ref = None
        for angle in angles:
            Rt, tb0, tmask, lost = tilted[angle]
            t0 = time.perf_counter()
            reg = A.register_to_template(
                tb0,
                affine,
                tmask,
                template=tmpl,
                template_mask=tmpl_mask,
                template_affine=tmpl_aff,
                pipeline=REG_VARIANTS[variant],
            )
            secs = time.perf_counter() - t0
            fa = reg["final_affine"]
            R = A.template_rotation(fa)
            pts = (fa @ mni_h.T).T[:, :3]
            if angle == 0.0:
                ref = (R, pts)
            R0, pts0 = ref
            # Anatomy rotated by Rt about the centre: ideal R = Rt R0 and ROI
            # centres at Rt (p0 - c) + c.
            rot_err = A.rotation_angle(R.T @ Rt @ R0)
            expect = (Rt @ (pts0 - centre).T).T + centre
            dist = np.linalg.norm(pts - expect, axis=1)
            sv = np.linalg.svd(fa[:3, :3], compute_uv=False)
            row = {
                "variant": variant,
                "axis": args.axis,
                "angle": angle,
                "fov_lost_pct": round(100 * lost, 2),
                "secs": round(secs, 1),
                "rot_err_deg": round(rot_err, 2),
                "roi_err_max_mm": round(float(dist.max()), 2),
                "tilt_measured": round(A.frame_tilt(R.T @ M)[0], 2),
                "sv_min": round(float(sv.min()), 3),
                "sv_max": round(float(sv.max()), 3),
            }
            rows.append(row)
            print(", ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    path = RESULTS / args.dataset / "regdiag" / "regdiag.csv"
    write_rows(rows, path)
    print(f"wrote {path}")


def _dice(a, b):
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def cmd_sensitivity(args):
    """Perturb each soft-score centre and width; rerun the ROI step only."""
    data, affine, bvals, bvecs = load_dataset(args.dataset)
    fit, mask, registration, _ = A.fit_and_register(data, affine, bvals, bvecs)
    del data
    registration["coords"] = A.native_template_coords(registration, mask.shape, affine)
    tensor, evecs, evals = fit.quadratic_form, fit.evecs, fit.evals

    def run(scores):
        return A.compute_alps(
            tensor,
            evecs,
            evals,
            mask,
            affine,
            registration,
            roi_methods=("data",),
            find_kwargs={"scores": scores},
        )

    base = run({})
    perturbations = [(None, None, 1.0)] + [
        (name, part, factor)
        for name in A.SOFT_SCORES
        for part in ("centre", "width")
        for factor in args.factors
    ]
    rows = []
    for name, part, factor in perturbations:
        if name is None:
            out, scores = base, {}
        else:
            centre, width = A.SOFT_SCORES[name]
            if part == "centre":
                scores = {name: (centre * factor, width)}
            else:
                scores = {name: (centre, width * factor)}
            out = run(scores)
        for frame in ("bvec", "template"):
            res, ref = out["results"][("data", frame)], base["results"][("data", frame)]
            key = f"data-{frame}"
            dice = [
                _dice(out["rois"][key][k], base["rois"][key][k])
                for k in base["rois"][key]
            ]
            fallbacks = sum(h["fallback"] for h in out["info"][key].values())
            row = {
                "param": name or "baseline",
                "part": part or "",
                "factor": factor,
                "value": "" if name is None else scores[name][int(part == "width")],
                "frame": frame,
                "alps_L": res["L"]["alps"],
                "alps_R": res["R"]["alps"],
                "alps_mean": res["mean"],
                "delta_pct": 100 * (res["mean"] / ref["mean"] - 1),
                "delta_L_pct": 100 * (res["L"]["alps"] / ref["L"]["alps"] - 1),
                "delta_R_pct": 100 * (res["R"]["alps"] / ref["R"]["alps"] - 1),
                "dice_mean": float(np.mean(dice)),
                "dice_min": float(np.min(dice)),
                "fallbacks": fallbacks,
            }
            rows.append(row)
            if frame == "template":
                print(
                    f"{row['param']:8s} {row['part']:6s} x{factor:<4g} "
                    f"ALPS {row['alps_mean']:.3f} ({row['delta_pct']:+.2f}%) "
                    f"dice {row['dice_mean']:.2f} fallback {fallbacks}",
                    flush=True,
                )
    path = RESULTS / args.dataset / "sensitivity" / "sensitivity.csv"
    write_rows(rows, path)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _fmt(x):
    return "NaN" if math.isnan(x) else f"{x:.3f}"


def collect():
    """Gather ALPS and registration results of every dataset on disk."""
    alps_rows, reg_rows = [], []
    for name in DATASETS:
        base = RESULTS / name
        direct = base / "direct" / "alps.csv"
        if direct.exists():
            meta = json.loads((base / "direct" / "alps.json").read_text())
            per = {}
            for r in read_rows(direct):
                key = (r["method"], r["frame"])
                v = per.setdefault(
                    key,
                    {
                        "mean": _f(r["alps_mean"]),
                        "mean_u": _f(r.get("alps_mean_unweighted")),
                    },
                )
                v[r["hemisphere"]] = _f(r["alps"])
            for (method, frame), v in per.items():
                alps_rows.append(
                    {
                        "dataset": name,
                        "experiment": "direct",
                        "axis": "",
                        "angle": "",
                        "tilt_measured": meta["frames"]["tilt_deg"],
                        "method": method,
                        "frame": frame,
                        "alps_L": v.get("L", math.nan),
                        "alps_R": v.get("R", math.nan),
                        "alps_mean": v["mean"],
                        "alps_mean_unweighted": v["mean_u"],
                    }
                )
        tilt = base / "tilt" / "tilt.csv"
        if tilt.exists():
            alps_rows.extend(
                {"dataset": name, "experiment": "tilt", **r} for r in read_rows(tilt)
            )
        reg = base / "regdiag" / "regdiag.csv"
        if reg.exists():
            reg_rows += [{"dataset": name, **r} for r in read_rows(reg)]
    return alps_rows, reg_rows


def collect_sensitivity():
    rows = []
    for name in DATASETS:
        path = RESULTS / name / "sensitivity" / "sensitivity.csv"
        if path.exists():
            rows += [{"dataset": name, **r} for r in read_rows(path)]
    return rows


def sensitivity_markdown(rows):
    """One table per dataset: template-frame ALPS change per perturbation."""
    lines = []
    for name in DATASETS:
        rs = [r for r in rows if r["dataset"] == name and r["frame"] == "template"]
        if not rs:
            continue
        base = next(r for r in rs if r["param"] == "baseline")
        factors = sorted({_f(r["factor"]) for r in rs if r["param"] != "baseline"})
        lines += [
            f"## Soft-score sensitivity: {name} (`data`, template frame)",
            "",
            f"Baseline ALPS {_fmt(_f(base['alps_mean']))} "
            f"(L {_fmt(_f(base['alps_L']))}, R {_fmt(_f(base['alps_R']))}). "
            "Cells: change of the mean ALPS in % (smallest ROI Dice against the "
            "baseline; F = number of hemispheres that fell back).",
            "",
            "| score | part | " + " | ".join(f"x{f:g}" for f in factors) + " |",
            "|---|---|" + "---|" * len(factors),
        ]
        table = {(r["param"], r["part"], _f(r["factor"])): r for r in rs}
        for param in A.SOFT_SCORES:
            for part in ("centre", "width"):
                cells = []
                for f in factors:
                    r = table.get((param, part, f))
                    if r is None:
                        cells.append("—")
                        continue
                    cell = f"{_f(r['delta_pct']):+.1f} ({_f(r['dice_min']):.2f})"
                    if int(r["fallbacks"]):
                        cell += f" F{r['fallbacks']}"
                    cells.append(cell)
                lines.append(f"| {param} | {part} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def markdown(alps_rows, reg_rows, *, sens_rows=()):
    cols = [
        (method, frame) for method in A.ROI_METHODS for frame in ("bvec", "template")
    ]
    head = " | ".join(f"{m} {f}" for m, f in cols)
    lines = ["<!-- generated by `python alps/experiments.py summary` -->", ""]

    direct = [r for r in alps_rows if r["experiment"] == "direct"]
    if direct:
        lines += [
            "## ALPS index (direct runs, mean of L and R; L/R in brackets)",
            "",
            "| dataset | measured tilt (deg) | " + head + " |",
            "|---|---|" + "---|" * len(cols),
        ]
        for name in DATASETS:
            rs = {(r["method"], r["frame"]): r for r in direct if r["dataset"] == name}
            if not rs:
                continue
            cells = []
            for key in cols:
                r = rs.get(key)
                cells.append(
                    "—"
                    if r is None
                    else f"{_fmt(_f(r['alps_mean']))} "
                    f"({_fmt(_f(r['alps_L']))}/{_fmt(_f(r['alps_R']))})"
                )
            tilt = _f(next(iter(rs.values()))["tilt_measured"])
            lines.append(f"| {name} | {tilt:.2f} | " + " | ".join(cells) + " |")
        lines.append("")

    for name in DATASETS:
        rs = [
            r for r in alps_rows if r["experiment"] == "tilt" and r["dataset"] == name
        ]
        if not rs:
            continue
        axis = rs[0]["axis"]
        angles = sorted({_f(r["angle"]) for r in rs}, key=lambda a: (abs(a), -a))
        table = {(_f(r["angle"]), r["method"], r["frame"]): r for r in rs}
        lines += [
            f"## Tilt series: {name} (about {axis}; ALPS mean of L and R)",
            "",
            "| tilt (deg) | measured tilt (deg) | " + head + " |",
            "|---|---|" + "---|" * len(cols),
        ]
        for a in angles:
            meas = next(_f(r["tilt_measured"]) for r in rs if _f(r["angle"]) == a)
            cells = [
                _fmt(_f(table[(a, *k)]["alps_mean"])) if (a, *k) in table else "—"
                for k in cols
            ]
            lines.append(f"| {a:+.1f} | {meas:.2f} | " + " | ".join(cells) + " |")
        stats = []
        for k in cols:
            v = np.array(
                [_f(table[(a, *k)]["alps_mean"]) for a in angles if (a, *k) in table]
            )
            ok = v[~np.isnan(v)]
            s = f"{100 * ok.std() / ok.mean():.1f}%" if ok.size > 1 else "—"
            if np.isnan(v).any():
                s += f" ({int(np.isnan(v).sum())} NaN)"
            stats.append(s)
        lines += ["| CV across angles | | " + " | ".join(stats) + " |", ""]

    if reg_rows:
        lines += [
            "## Template registration under tilt (skull-stripped b0 and template)",
            "",
            "Rotation error: angle of `R^T Rt R0`; ROI error: largest distance between",
            "the mapped MNI ROI centres and the 0-deg centres rotated by the tilt.",
            "",
            "| dataset | variant | tilt (deg) | rotation error (deg) | "
            "max ROI error (mm) | singular values | time (s) |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in reg_rows:
            lines.append(
                f"| {r['dataset']} | {r['variant']} | {_f(r['angle']):+.1f} | "
                f"{r['rot_err_deg']} | {r['roi_err_max_mm']} | "
                f"{r['sv_min']}–{r['sv_max']} | {r['secs']} |"
            )
        lines.append("")
    lines += sensitivity_markdown(sens_rows)
    return "\n".join(lines)


def cmd_summary(args):
    alps_rows, reg_rows = collect()
    sens_rows = collect_sensitivity()
    if sens_rows:
        write_rows(sens_rows, RESULTS / "summary_sensitivity.csv")
    if alps_rows:
        write_rows(alps_rows, RESULTS / "summary_alps.csv")
    if reg_rows:
        write_rows(reg_rows, RESULTS / "summary_regdiag.csv")
    text = markdown(alps_rows, reg_rows, sens_rows=sens_rows)
    (RESULTS / "tables.md").write_text(text)
    print(text)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--results-dir", default="alps_results", help="Root of the results tree."
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_dataset(sp):
        sp.add_argument("--dataset", choices=list(DATASETS), required=True)

    sp = sub.add_parser("direct", help="Run alps.py on a dataset.")
    add_dataset(sp)
    sp.set_defaults(func=cmd_direct)

    sp = sub.add_parser("tilt", help="Tilt series of the full pipeline.")
    add_dataset(sp)
    sp.add_argument("--angles", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    sp.add_argument("--axis", choices=["x", "y", "z"], default="x")
    sp.add_argument("--keep-dwi", action="store_true", help="Keep tilted DWIs.")
    sp.set_defaults(func=cmd_tilt)

    sp = sub.add_parser("regdiag", help="Registration accuracy under tilt.")
    add_dataset(sp)
    sp.add_argument("--angles", type=float, nargs="+", default=[0.0, 5.0, 10.0, -10.0])
    sp.add_argument("--axis", choices=["x", "y", "z"], default="x")
    sp.add_argument(
        "--variants", nargs="+", choices=list(REG_VARIANTS), default=list(REG_VARIANTS)
    )
    sp.set_defaults(func=cmd_regdiag)

    sp = sub.add_parser(
        "sensitivity", help="Soft-score sensitivity of the data method."
    )
    add_dataset(sp)
    sp.add_argument("--factors", type=float, nargs="+", default=[0.8, 0.9, 1.1, 1.2])
    sp.set_defaults(func=cmd_sensitivity)

    sp = sub.add_parser("summary", help="Collect results into tables.")
    sp.set_defaults(func=cmd_summary)
    return p


def main(*, argv=None):
    global RESULTS
    args = build_parser().parse_args(argv)
    RESULTS = Path(args.results_dir)
    args.func(args)


if __name__ == "__main__":
    main()
