"""Rotation-corrected DTI-ALPS index with standard and data-driven ROIs.

The DTI-ALPS index (diffusion tensor image analysis along the perivascular
space) compares diffusivity along the x-axis (left-right, the assumed
direction of the medullary veins and their perivascular spaces) with the
diffusivity perpendicular to it, in projection and association fibre regions
next to the bodies of the lateral ventricles [1]_:

.. math::

    ALPS = \\frac{mean(D_{xx,proj}, D_{xx,assoc})}{mean(D_{yy,proj}, D_{zz,assoc})}

The index assumes that the tensor axes are the anatomical axes. A tensor
fitted with gradient directions in the scanner (voxel) frame breaks that
assumption whenever the head is tilted. This script rotates the tensors into
the frame of the MNI template before it reads the diagonal elements. It
reports the index in both frames:

- ``bvec``: the tensor as fitted, as in most published pipelines.
- ``template``: the tensor rotated into the template-aligned frame.

ROIs come from one of two methods:

- ``sphere``: 5 mm spheres at published MNI coordinates [2]_, mapped to
  native space.
- ``data``: a constrained data-driven search. It looks lateral to the
  lateral-ventricle body for coherent projection (z-oriented) and association
  (y-oriented) fibres. It rejects voxels with low linearity or high planarity,
  which removes crossing fibres, the main confound of the index.

The script is self-contained. Its functions are written so that they can
move to ``dipy/stats/alps.py`` and its ``main`` to a ``Workflow``.

References
----------
.. [1] Taoka T, et al. Evaluation of glymphatic system activity with the
   diffusion MR technique: diffusion tensor image analysis along the
   perivascular space (DTI-ALPS) in Alzheimer's disease cases. Jpn J Radiol.
   2017;35:172-178.
.. [2] Park CJ, et al. Association of diffusion along the perivascular spaces
   with clinical symptoms and dopaminergic degeneration in Parkinson's
   disease. npj Parkinsons Dis. 2026. doi:10.1038/s41531-026-01620-4
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.linalg import polar

logger = logging.getLogger("alps")

#: ROI centres (MNI mm) used by Park et al. 2026 (after Zhang et al. 2021).
ALPS_MNI_COORDS = {
    "L": {"proj": (-28.0, -12.0, 24.0), "assoc": (-40.0, -12.0, 24.0)},
    "R": {"proj": (24.0, -12.0, 24.0), "assoc": (36.0, -12.0, 24.0)},
}

#: Index of the principal-direction axis for each fibre group (x=0, y=1, z=2).
FIBRE_AXIS = {"proj": 2, "assoc": 1}

#: Integer labels for the ROI label image.
ROI_LABELS = {("L", "proj"): 1, ("L", "assoc"): 2, ("R", "proj"): 3, ("R", "assoc"): 4}

HEMISPHERE_SIGN = {"L": -1.0, "R": 1.0}


# ---------------------------------------------------------------------------
# Frames and rotations
# ---------------------------------------------------------------------------


def bvec_frame_to_world(affine, *, convention="fsl"):
    """Return the rotation that maps the gradient (bvec) frame to world axes.

    Parameters
    ----------
    affine : ndarray (4, 4)
        Voxel-to-world affine of the DWI.
    convention : {"fsl", "voxel", "world"}, optional
        Frame of the b-vectors.

        - ``"fsl"``: voxel axes, with the first axis flipped when the
          determinant of the affine is positive (FSL and dcm2niix convention).
        - ``"voxel"``: voxel axes, no flip.
        - ``"world"``: already in world (RAS+) axes.

    Returns
    -------
    M : ndarray (3, 3)
        Orthogonal matrix such that ``v_world = M @ v_bvec``.
    """
    if convention == "world":
        return np.eye(3)
    linear = np.asarray(affine, dtype=float)[:3, :3]
    U = polar(linear)[0]
    if convention == "voxel":
        return U
    if convention == "fsl":
        F = np.eye(3)
        if np.linalg.det(linear) > 0:
            F[0, 0] = -1.0
        return U @ F
    raise ValueError(f"Unknown b-vector convention: {convention!r}")


def template_rotation(final_affine):
    """Rotation part of a template-to-native affine.

    Parameters
    ----------
    final_affine : ndarray (4, 4)
        Affine that maps template world coordinates to native world
        coordinates, as returned by :func:`dipy.align.affine_registration`
        with the template as the static image.

    Returns
    -------
    R : ndarray (3, 3)
        Closest rotation (polar decomposition), such that a direction ``u`` in
        the template frame is ``R @ u`` in the native world frame.
    """
    return polar(np.asarray(final_affine, dtype=float)[:3, :3])[0]


def rotation_angle(R):
    """Angle (degrees) of the rotation ``R``."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def frame_tilt(Q):
    """Angle (degrees) between two frames, ignoring axis sign flips.

    Axis flips (such as the FSL x flip) do not change the diagonal tensor
    elements, so the tilt that matters for ALPS is the angle of ``Q S``, where
    ``S`` flips the axes whose diagonal entry in ``Q`` is negative.

    Returns
    -------
    total : float
        Rotation angle of ``Q S``.
    per_axis : list of float
        Angle between each axis of one frame and the same axis of the other.
    """
    signs = np.where(np.diag(Q) < 0, -1.0, 1.0)
    QS = Q @ np.diag(signs)
    per_axis = np.degrees(np.arccos(np.clip(np.abs(np.diag(Q)), 0.0, 1.0)))
    total = rotation_angle(QS) if np.linalg.det(QS) > 0 else float("nan")
    return total, per_axis.tolist()


def rotate_tensor(tensor, Q):
    """Rotate a field of tensors: ``D' = Q D Q^T``.

    Parameters
    ----------
    tensor : ndarray (..., 3, 3)
        Diffusion tensors.
    Q : ndarray (3, 3)
        Rotation from the frame of ``tensor`` to the target frame.

    Returns
    -------
    ndarray (..., 3, 3)
    """
    return np.einsum("ij,...jk,lk->...il", Q, tensor, Q)


def rotate_vectors(vectors, Q):
    """Rotate eigenvector matrices whose columns are vectors: ``V' = Q V``."""
    return np.einsum("ij,...jk->...ik", Q, vectors)


# ---------------------------------------------------------------------------
# Registration and coordinates
# ---------------------------------------------------------------------------


def load_template(*, voxel_size=2.0):
    """Return the MNI ICBM 2009a T2 template, a brain mask and the affine.

    The brain mask is the MNI ICBM 2009c mask that ships with the 2009a
    template. Both images are resampled to an isotropic grid of
    ``voxel_size`` mm (the native files are 1 mm).

    Returns
    -------
    template : ndarray (X, Y, Z)
    template_mask : ndarray of bool (X, Y, Z)
    template_affine : ndarray (4, 4)
    """
    import nibabel as nib

    from dipy.align.imaffine import AffineMap
    from dipy.data import read_mni_template

    t2 = read_mni_template(version="a", contrast="T2")
    mask_path = (
        Path(t2.get_filename()).parent / "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"
    )
    mask = nib.load(mask_path)

    zoom = voxel_size / np.abs(np.diag(t2.affine)[:3])
    shape = tuple(int(np.ceil(s / z)) for s, z in zip(t2.shape, zoom))
    aff = t2.affine.copy()
    aff[:3, :3] = t2.affine[:3, :3] * zoom
    out = []
    for img, interp in ((t2, "linear"), (mask, "nearest")):
        amap = AffineMap(
            np.eye(4),
            domain_grid_shape=shape,
            domain_grid2world=aff,
            codomain_grid_shape=img.shape,
            codomain_grid2world=img.affine,
        )
        out.append(
            amap.transform(np.asarray(img.get_fdata(), float), interpolation=interp)
        )
    return out[0], out[1] > 0.5, aff


def register_to_template(
    b0,
    affine,
    mask,
    *,
    template=None,
    template_mask=None,
    template_affine=None,
    method="affine",
    pipeline=("center_of_mass", "translation", "rigid", "affine"),
    level_iters=(200, 50, 10),
):
    """Register a skull-stripped mean b0 image to a skull-stripped template.

    Both images are multiplied by their brain masks before registration.
    ``experiments.py regdiag`` measures the accuracy under head tilt.

    Parameters
    ----------
    b0 : ndarray (X, Y, Z)
        Mean b0 image.
    affine : ndarray (4, 4)
        Voxel-to-world affine of ``b0``.
    mask : ndarray of bool (X, Y, Z)
        Brain mask of ``b0``.
    template, template_mask, template_affine : ndarray, optional
        Template image, its brain mask and its affine. Default: MNI ICBM
        2009a T2 with the 2009c brain mask at 2 mm (:func:`load_template`).
    method : {"affine", "syn"}, optional
        ``"syn"`` runs SyN after the affine step. The rotation always comes
        from the affine step.
    pipeline : sequence of str, optional
        Steps of :func:`dipy.align.affine_registration`. End at ``"rigid"``
        for a rigid-only registration (diagnostics only).
    level_iters : sequence of int, optional
        Iterations per resolution level of the affine registration.

    Returns
    -------
    dict
        ``final_affine`` (template world -> native world), ``mapping``
        (an ``AffineMap`` or ``DiffeomorphicMap`` with the template as the
        domain), ``template`` (not skull-stripped) and ``template_affine``.
    """
    from dipy.align import affine_registration, syn_registration
    from dipy.align.imaffine import AffineMap

    if template is None:
        template, template_mask, template_affine = load_template()
    if template_mask is None:
        template_mask = np.ones(template.shape, bool)

    moving = b0 * mask
    static = template * template_mask
    _, final_affine = affine_registration(
        moving,
        static,
        moving_affine=affine,
        static_affine=template_affine,
        pipeline=list(pipeline),
        metric="MI",
        level_iters=list(level_iters),
    )
    if method == "affine":
        mapping = AffineMap(
            final_affine,
            domain_grid_shape=template.shape,
            domain_grid2world=template_affine,
            codomain_grid_shape=b0.shape,
            codomain_grid2world=affine,
        )
    elif method == "syn":
        _, mapping = syn_registration(
            moving,
            static,
            moving_affine=affine,
            static_affine=template_affine,
            prealign=final_affine,
        )
    else:
        raise ValueError(f"Unknown registration method: {method!r}")

    return {
        "final_affine": final_affine,
        "mapping": mapping,
        "template": template,
        "template_affine": template_affine,
    }


def native_template_coords(registration, native_shape, native_affine):
    """Template world coordinates (mm) of every native voxel.

    Parameters
    ----------
    registration : dict
        Output of :func:`register_to_template`.
    native_shape : tuple
        Shape of the native grid.
    native_affine : ndarray (4, 4)
        Voxel-to-world affine of the native grid.

    Returns
    -------
    coords : ndarray (X, Y, Z, 3)
        Template (MNI) coordinates in mm.
    """
    from dipy.align.imaffine import AffineMap

    mapping = registration["mapping"]
    if isinstance(mapping, AffineMap):
        to_template = np.linalg.inv(registration["final_affine"]) @ native_affine
        ijk = np.indices(native_shape, dtype=float).reshape(3, -1)
        xyz = to_template[:3, :3] @ ijk + to_template[:3, 3:4]
        return xyz.T.reshape(*native_shape, 3)

    template_shape = registration["template"].shape
    t_aff = registration["template_affine"]
    ijk = np.indices(template_shape, dtype=float).reshape(3, -1)
    grid = (t_aff[:3, :3] @ ijk + t_aff[:3, 3:4]).reshape(3, *template_shape)
    return np.stack(
        [mapping.transform_inverse(grid[k], interpolation="linear") for k in range(3)],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Tensor-derived maps
# ---------------------------------------------------------------------------


def eigen_maps(evals):
    """FA, MD, linearity and planarity from eigenvalues sorted descending.

    Linearity is ``(l1 - l2) / l1`` and planarity is ``(l2 - l3) / l1``
    (Westin et al. 1997).
    """
    from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity

    fa = np.nan_to_num(fractional_anisotropy(evals))
    md = mean_diffusivity(evals)
    l1 = evals[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        cl = np.where(l1 > 0, (evals[..., 0] - evals[..., 1]) / l1, 0.0)
        cp = np.where(l1 > 0, (evals[..., 1] - evals[..., 2]) / l1, 0.0)
    return {"fa": fa, "md": md, "cl": cl, "cp": cp}


# ---------------------------------------------------------------------------
# ROIs
# ---------------------------------------------------------------------------


def sphere_rois(coords, mask, *, centres=None, radius=2.5):
    """Spherical ROIs at template coordinates, on the native grid.

    Parameters
    ----------
    coords : ndarray (X, Y, Z, 3)
        Template coordinates of each native voxel.
    mask : ndarray (X, Y, Z) of bool
        Brain mask.
    centres : dict, optional
        ``{hemisphere: {group: (x, y, z)}}``. Default: :data:`ALPS_MNI_COORDS`.
    radius : float, optional
        Sphere radius in mm. Default 2.5 (5 mm diameter).

    Returns
    -------
    dict
        ``{(hemisphere, group): bool mask}``.
    """
    centres = centres or ALPS_MNI_COORDS
    rois = {}
    for hemi, groups in centres.items():
        for group, centre in groups.items():
            dist = np.linalg.norm(coords - np.asarray(centre), axis=-1)
            rois[(hemi, group)] = mask & (dist <= radius)
    return rois


def ventricle_mask(
    maps,
    coords,
    mask,
    *,
    md_min=2.0e-3,
    fa_max=0.2,
    min_volume_mm3=500.0,
    voxel_volume=1.0,
):
    """Lateral-ventricle mask from CSF-like voxels near the midline.

    Voxels with high MD and low FA inside a central box (template mm:
    ``|x| < 30``, ``-45 <= y <= 25``, ``0 <= z <= 40``) are labelled; connected
    components smaller than ``min_volume_mm3`` are dropped.
    """
    X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]
    box = (np.abs(X) < 30) & (Y >= -45) & (Y <= 25) & (Z >= 0) & (Z <= 40)
    csf = mask & box & (maps["md"] > md_min) & (maps["fa"] < fa_max)
    labels, n = ndimage.label(csf)
    if n == 0:
        return csf
    sizes = ndimage.sum(csf, labels, index=np.arange(1, n + 1)) * voxel_volume
    keep = np.flatnonzero(sizes >= min_volume_mm3) + 1
    return np.isin(labels, keep)


def find_alps_rois(
    evecs,
    maps,
    coords,
    mask,
    *,
    voxel_size=(2.0, 2.0, 2.0),
    radius=2.5,
    y_range=(-24.0, 0.0),
    z_range=(16.0, 32.0),
    x_min=18.0,
    x_span=30.0,
    fa_min=0.3,
    cl_min=0.4,
    cp_max=0.3,
    max_angle=25.0,
    min_voxels=2,
    ventricles=None,
):
    """Constrained data-driven search for ALPS ROIs.

    For each hemisphere the search runs inside a prior box in template
    coordinates. When a ventricle mask is given, the box starts at the
    lateral wall of the ventricle body and its z range is limited to the
    levels where the ventricle body exists. Candidate voxels have FA, linearity
    and planarity within thresholds, and a principal eigenvector within
    ``max_angle`` of z (projection) or y (association). The ROI centre is the
    (y, z) position whose disc of radius ``radius`` holds the most voxels of
    the smaller group, with projection fibres medial to association fibres.
    Each ROI keeps the candidates in that disc within ``radius`` (mm, along x)
    of the group's median x.

    Parameters
    ----------
    evecs : ndarray (X, Y, Z, 3, 3)
        Eigenvectors (columns) in the frame to evaluate.
    maps : dict
        Output of :func:`eigen_maps`.
    coords : ndarray (X, Y, Z, 3)
        Template coordinates of each native voxel.
    mask : ndarray (X, Y, Z) of bool
        Brain mask.
    ventricles : ndarray (X, Y, Z) of bool, optional
        Output of :func:`ventricle_mask`.

    Returns
    -------
    rois : dict
        ``{(hemisphere, group): bool mask}``.
    info : dict
        Per hemisphere: chosen centre, box limits and candidate counts.
    """
    X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]
    e1 = evecs[..., :, 0]
    cos_t = np.cos(np.radians(max_angle))
    quality = (
        mask & (maps["fa"] >= fa_min) & (maps["cl"] >= cl_min) & (maps["cp"] <= cp_max)
    )
    step = float(min(voxel_size))

    rois, info = {}, {}
    for hemi, sign in HEMISPHERE_SIGN.items():
        sx = sign * X
        lo_x, z_lo, z_hi = x_min, z_range[0], z_range[1]
        if ventricles is not None:
            vent = ventricles & (sx > 0) & (Y >= y_range[0]) & (Y <= y_range[1])
            vent_z = vent & (Z >= z_range[0]) & (Z <= z_range[1])
            if vent_z.any():
                lo_x = float(np.percentile(sx[vent_z], 99))
                z_lo = max(z_range[0], float(Z[vent_z].min()))
                z_hi = min(z_range[1], float(Z[vent_z].max()))
        hi_x = lo_x + x_span
        box = (sx >= lo_x) & (sx <= hi_x) & (Y >= y_range[0]) & (Y <= y_range[1])
        box &= (Z >= z_lo) & (Z <= z_hi)
        if ventricles is not None:
            box &= ~ventricles

        cand = {
            "proj": box & quality & (np.abs(e1[..., FIBRE_AXIS["proj"]]) >= cos_t),
            "assoc": box & quality & (np.abs(e1[..., FIBRE_AXIS["assoc"]]) >= cos_t),
        }
        pts = {g: np.argwhere(c) for g, c in cand.items()}
        yz = {g: np.column_stack([Y[c], Z[c]]) for g, c in cand.items()}
        xs = {g: sx[c] for g, c in cand.items()}

        best, best_score = None, (-1, -np.inf)
        for y0 in np.arange(y_range[0], y_range[1] + 1e-6, step):
            for z0 in np.arange(z_lo, z_hi + 1e-6, step):
                sel = {
                    g: np.sum((yz[g] - (y0, z0)) ** 2, axis=1) <= radius**2
                    for g in cand
                }
                n_p, n_a = int(sel["proj"].sum()), int(sel["assoc"].sum())
                if min(n_p, n_a) < min_voxels:
                    continue
                if np.median(xs["proj"][sel["proj"]]) >= np.median(
                    xs["assoc"][sel["assoc"]]
                ):
                    continue
                # Tie-break: prefer centres close to the published coordinates.
                ref = ALPS_MNI_COORDS[hemi]["proj"]
                score = (min(n_p, n_a), -np.hypot(y0 - ref[1], z0 - ref[2]))
                if score > best_score:
                    best, best_score = (y0, z0, sel), score

        hemi_info = {
            "x_range": [lo_x, hi_x],
            "z_range": [z_lo, z_hi],
            "n_candidates": {g: int(c.sum()) for g, c in cand.items()},
            "centre_yz": None,
        }
        for group in cand:
            roi = np.zeros(mask.shape, dtype=bool)
            if best is not None:
                sel = best[2][group]
                x_med = np.median(xs[group][sel])
                keep = sel & (np.abs(xs[group] - x_med) <= radius)
                idx = pts[group][keep]
                roi[tuple(idx.T)] = True
            rois[(hemi, group)] = roi
        if best is not None:
            hemi_info["centre_yz"] = [float(best[0]), float(best[1])]
        else:
            logger.warning("No data-driven ALPS ROI found in hemisphere %s", hemi)
        info[hemi] = hemi_info
    return rois, info


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def roi_metrics(tensor, evecs, maps, roi, group):
    """Mean diagonal diffusivities and quality metrics in one ROI."""
    n = int(roi.sum())
    out = {"nvox": n}
    if n == 0:
        for key in ("Dxx", "Dyy", "Dzz", "fa", "md", "cl", "cp", "angle_deg"):
            out[key] = np.nan
        return out
    mean_tensor = tensor[roi].mean(axis=0)
    out["Dxx"], out["Dyy"], out["Dzz"] = np.diag(mean_tensor)
    for key in ("fa", "md", "cl", "cp"):
        out[key] = float(maps[key][roi].mean())
    cos_a = np.clip(np.abs(evecs[roi][:, FIBRE_AXIS[group], 0]), 0.0, 1.0)
    out["angle_deg"] = float(np.degrees(np.arccos(cos_a)).mean())
    return out


def alps_index(tensor, evecs, maps, rois):
    """ALPS index per hemisphere and the metrics of each ROI.

    Parameters
    ----------
    tensor : ndarray (X, Y, Z, 3, 3)
        Tensors in the frame to evaluate.
    evecs : ndarray (X, Y, Z, 3, 3)
        Eigenvectors (columns) in the same frame.
    maps : dict
        Output of :func:`eigen_maps`.
    rois : dict
        ``{(hemisphere, group): bool mask}``.

    Returns
    -------
    dict
        ``{hemisphere: {"alps": float, "proj": {...}, "assoc": {...}}}`` and
        ``"mean"``: the mean of the hemispheric indices.
    """
    result = {}
    for hemi in HEMISPHERE_SIGN:
        proj = roi_metrics(tensor, evecs, maps, rois[(hemi, "proj")], "proj")
        assoc = roi_metrics(tensor, evecs, maps, rois[(hemi, "assoc")], "assoc")
        num = np.mean([proj["Dxx"], assoc["Dxx"]])
        den = np.mean([proj["Dyy"], assoc["Dzz"]])
        result[hemi] = {"alps": float(num / den), "proj": proj, "assoc": assoc}
    result["mean"] = float(np.mean([result[h]["alps"] for h in HEMISPHERE_SIGN]))
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def compute_alps(
    tensor,
    evecs,
    evals,
    mask,
    affine,
    registration,
    *,
    bvec_convention="fsl",
    roi_methods=("sphere", "data"),
    find_kwargs=None,
):
    """Run both frames and the requested ROI methods.

    Parameters
    ----------
    tensor : ndarray (X, Y, Z, 3, 3)
        Tensors in the b-vector frame (as fitted).
    evecs, evals : ndarray
        Eigenvectors (columns) and eigenvalues (descending) of ``tensor``.
    mask : ndarray (X, Y, Z) of bool
        Brain mask.
    affine : ndarray (4, 4)
        Voxel-to-world affine of the DWI.
    registration : dict
        Output of :func:`register_to_template`, or any dict with
        ``final_affine`` and a ``coords`` array of template coordinates.

    Returns
    -------
    dict
        ``results[(method, frame)]`` from :func:`alps_index`, ``rois``,
        ``frames`` (rotations) and ``info``.
    """
    find_kwargs = dict(find_kwargs or {})
    M = bvec_frame_to_world(affine, convention=bvec_convention)
    R = template_rotation(registration["final_affine"])
    Q = R.T @ M

    coords = registration.get("coords")
    if coords is None:
        coords = native_template_coords(registration, mask.shape, affine)
    valid = mask & np.all(np.isfinite(coords), axis=-1)

    maps = eigen_maps(evals)
    voxel_size = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))
    ventricles = ventricle_mask(
        maps, coords, valid, voxel_volume=float(np.prod(voxel_size))
    )
    frames = {
        "bvec": (tensor, evecs),
        "template": (rotate_tensor(tensor, Q), rotate_vectors(evecs, Q)),
    }

    results, rois, info = {}, {}, {}
    if "sphere" in roi_methods:
        rois["sphere"] = sphere_rois(coords, valid)
    for frame, (D, V) in frames.items():
        if "data" in roi_methods:
            key = f"data-{frame}"
            rois[key], info[key] = find_alps_rois(
                V,
                maps,
                coords,
                valid,
                voxel_size=voxel_size,
                ventricles=ventricles,
                **find_kwargs,
            )
            results[("data", frame)] = alps_index(D, V, maps, rois[key])
        if "sphere" in roi_methods:
            results[("sphere", frame)] = alps_index(D, V, maps, rois["sphere"])

    return {
        "results": results,
        "rois": rois,
        "info": info,
        "maps": maps,
        "ventricles": ventricles,
        "coords": coords,
        "frames": {
            "M": M,
            "R": R,
            "Q": Q,
            "tilt_deg": frame_tilt(Q)[0],
            "tilt_per_axis_deg": frame_tilt(Q)[1],
        },
        "evecs_template": frames["template"][1],
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

CSV_FIELDS = ["method", "frame", "hemisphere", "alps", "alps_mean"] + [
    f"{group}_{key}"
    for group in ("proj", "assoc")
    for key in ("Dxx", "Dyy", "Dzz", "nvox", "fa", "md", "cl", "cp", "angle_deg")
]


def results_rows(results):
    """Flatten :func:`compute_alps` results into CSV rows."""
    rows = []
    for (method, frame), res in results.items():
        for hemi in HEMISPHERE_SIGN:
            row = {
                "method": method,
                "frame": frame,
                "hemisphere": hemi,
                "alps": res[hemi]["alps"],
                "alps_mean": res["mean"],
            }
            for group in ("proj", "assoc"):
                for key, value in res[hemi][group].items():
                    row[f"{group}_{key}"] = value
            rows.append(row)
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def roi_label_image(rois):
    """Label image from an ROI dict (see :data:`ROI_LABELS`)."""
    shape = next(iter(rois.values())).shape
    labels = np.zeros(shape, dtype=np.uint8)
    for key, roi in rois.items():
        labels[roi] = ROI_LABELS[key]
    return labels


def plot_qc(out, b0, template_native, path):
    """Save a QC figure: ROIs on colour FA, and the registration check.

    Colour FA uses the template-frame eigenvectors (red = left-right,
    green = anterior-posterior, blue = superior-inferior in the template).
    ``template_native`` is the template resampled onto the native grid.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fa = out["maps"]["fa"]
    cfa = np.clip(np.abs(out["evecs_template"][..., :, 0]) * fa[..., None], 0, 1)
    roi_sets = {
        k: v for k, v in out["rois"].items() if k in ("sphere", "data-template")
    }
    any_roi = np.zeros(fa.shape, dtype=bool)
    for rois in roi_sets.values():
        for roi in rois.values():
            any_roi |= roi
    k = (
        int(np.median(np.argwhere(any_roi)[:, 2]))
        if any_roi.any()
        else fa.shape[2] // 2
    )

    colours = {"proj": "white", "assoc": "yellow"}
    n = len(roi_sets) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    for ax, (name, rois) in zip(axes, roi_sets.items()):
        ax.imshow(np.transpose(cfa[:, :, k], (1, 0, 2)), origin="lower")
        for (_hemi, group), roi in rois.items():
            sl = roi[:, :, k].T.astype(float)
            if sl.any():
                ax.contour(sl, levels=[0.5], colors=colours[group], linewidths=1)
        vent = out["ventricles"][:, :, k].T.astype(float)
        if vent.any():
            ax.contour(vent, levels=[0.5], colors="cyan", linewidths=0.5)
        ax.set_title(f"{name} ROIs (slice {k})")
        ax.axis("off")

    ax = axes[-1]
    ax.imshow(b0[:, :, k].T, origin="lower", cmap="gray")
    ax.contour(template_native[:, :, k].T, levels=4, colors="red", linewidths=0.5)
    ax.set_title(f"b0 + template contours (tilt {out['frames']['tilt_deg']:.1f} deg)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="Rotation-corrected DTI-ALPS index with sphere and "
        "data-driven ROIs."
    )
    p.add_argument("dwi", help="Preprocessed 4D DWI NIfTI.")
    p.add_argument("bval", help="b-values file.")
    p.add_argument("bvec", help="b-vectors file.")
    p.add_argument("--mask", help="Brain mask. Default: median_otsu on the b0s.")
    p.add_argument("--out-dir", default="alps_out", help="Output directory.")
    p.add_argument("--roi-method", choices=["sphere", "data", "both"], default="both")
    p.add_argument(
        "--bvec-convention", choices=["fsl", "voxel", "world"], default="fsl"
    )
    p.add_argument("--reg", choices=["affine", "syn"], default="affine")
    p.add_argument(
        "--max-bval",
        type=float,
        default=None,
        help="Drop volumes with b above this value (e.g. 1100 to keep b=1000).",
    )
    p.add_argument("--b0-threshold", type=float, default=50.0)
    p.add_argument("--no-qc", action="store_true", help="Skip the QC figure.")
    return p


def main(*, argv=None):
    import nibabel as nib

    from dipy.core.gradients import gradient_table
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti, save_nifti
    from dipy.reconst.dti import TensorModel
    from dipy.segment.mask import median_otsu

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, affine = load_nifti(args.dwi)
    bvals, bvecs = read_bvals_bvecs(args.bval, args.bvec)
    if args.max_bval is not None:
        keep = bvals <= args.max_bval
        data, bvals, bvecs = data[..., keep], bvals[keep], bvecs[keep]
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=args.b0_threshold)
    b0_idx = np.flatnonzero(gtab.b0s_mask)
    b0 = data[..., b0_idx].mean(axis=-1)

    if args.mask:
        mask = nib.load(args.mask).get_fdata() > 0
    else:
        _, mask = median_otsu(
            data, vol_idx=b0_idx, median_radius=4, numpass=4, dilate=2
        )
        save_nifti(out_dir / "mask.nii.gz", mask.astype(np.uint8), affine)
        logger.info("Computed brain mask with median_otsu")

    logger.info("Fitting DTI on %d volumes", data.shape[-1])
    fit = TensorModel(gtab).fit(data, mask=mask)

    logger.info("Registering mean b0 to the MNI template (%s)", args.reg)
    registration = register_to_template(b0, affine, mask, method=args.reg)

    methods = ("sphere", "data") if args.roi_method == "both" else (args.roi_method,)
    out = compute_alps(
        fit.quadratic_form,
        fit.evecs,
        fit.evals,
        mask,
        affine,
        registration,
        bvec_convention=args.bvec_convention,
        roi_methods=methods,
    )

    write_csv(results_rows(out["results"]), out_dir / "alps.csv")
    for name, rois in out["rois"].items():
        save_nifti(out_dir / f"rois_{name}.nii.gz", roi_label_image(rois), affine)
    frames = {k: np.asarray(v).tolist() for k, v in out["frames"].items()}
    frames["final_affine"] = np.asarray(registration["final_affine"]).tolist()
    meta = {"frames": frames, "data_driven": out["info"], "args": vars(args)}
    (out_dir / "alps.json").write_text(json.dumps(meta, indent=2))

    if not args.no_qc:
        template_native = registration["mapping"].transform_inverse(
            registration["template"], interpolation="linear"
        )
        plot_qc(out, b0, template_native, out_dir / "qc_alps.png")

    for (method, frame), res in out["results"].items():
        logger.info(
            "ALPS %-6s %-8s L=%.3f R=%.3f mean=%.3f",
            method,
            frame,
            res["L"]["alps"],
            res["R"]["alps"],
            res["mean"],
        )
    logger.info(
        "Template rotation of the b-vector frame: %.2f deg", out["frames"]["tilt_deg"]
    )
    return out


if __name__ == "__main__":
    main()
