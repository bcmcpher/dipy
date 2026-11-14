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

ROIs come from one of three methods:

- ``sphere``: 5 mm spheres at published MNI coordinates [2]_, mapped to
  native space.
- ``data``: a soft-weighted, data-driven search. It looks lateral to the
  lateral-ventricle body for coherent projection (z-oriented) and association
  (y-oriented) fibres. Each voxel gets a weight in [0, 1] from smooth scores of
  orientation, linearity, planarity, FA and MD; low linearity and high
  planarity (crossing fibres, the main confound of the index) give low
  weights. Each ROI grows from a seed to the voxel count of a ``sphere`` ROI,
  and the index is the ratio of weighted mean diffusivities.
- ``data-legacy``: the first data-driven search, with hard thresholds. Kept
  for comparison.

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
import itertools
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

#: ROI methods, in report order.
ROI_METHODS = ("sphere", "data-legacy", "data")


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


def _prior_box(sx, Y, Z, ventricles, *, x_min, x_span, y_range, z_range):
    """Per-hemisphere search box in template coordinates.

    ``sx`` is the template x multiplied by the hemisphere sign, so that
    lateral is positive. With a ventricle mask, the box starts at the lateral
    wall of the ventricle body and its z range is limited to the levels where
    the ventricle body exists; ventricle voxels are excluded.

    Returns
    -------
    box : ndarray of bool
    x_lim, z_lim : tuple of float
        The ``sx`` and z limits of the box.
    """
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
    return box, (lo_x, hi_x), (z_lo, z_hi)


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
        box, (lo_x, hi_x), (z_lo, z_hi) = _prior_box(
            sx,
            Y,
            Z,
            ventricles,
            x_min=x_min,
            x_span=x_span,
            y_range=y_range,
            z_range=z_range,
        )

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


#: Default (centre, width) of each soft score in :func:`soft_weights`.
#: The centres are the hard thresholds of :func:`find_alps_rois`.
SOFT_SCORES = {
    "angle": (25.0, 5.0),  # deg between e1 and the fibre axis, falling
    "fa": (0.3, 0.05),  # rising
    "cl": (0.4, 0.05),  # rising
    "cp": (0.3, 0.05),  # falling
    "md": (1.1e-3, 0.1e-3),  # mm^2/s, falling (CSF partial volume)
}


def _rising(v, centre, width):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-(v - centre) / width))


def soft_weights(
    e1, maps, coords, group, hemi, *, scores=None, prior_sigma=6.0, prior_weight=0.2
):
    """Soft voxel weight in [0, 1] for one fibre group and hemisphere.

    The weight is the product of a falling sigmoid of the angle between the
    principal eigenvector and the fibre axis, rising sigmoids of FA and
    linearity, falling sigmoids of planarity and MD, and a weak Gaussian prior
    around the published MNI centre of the ROI.

    Parameters
    ----------
    e1 : ndarray (X, Y, Z, 3)
        Principal eigenvectors in the frame to evaluate.
    maps : dict
        Output of :func:`eigen_maps`.
    coords : ndarray (X, Y, Z, 3)
        Template coordinates of each native voxel.
    group : {"proj", "assoc"}
    hemi : {"L", "R"}
    scores : dict, optional
        ``{name: (centre, width)}`` overrides of :data:`SOFT_SCORES`.
    prior_sigma : float, optional
        Standard deviation (template mm) of the prior around the MNI centre.
    prior_weight : float, optional
        Depth of the prior: the factor falls from 1 at the centre to
        ``1 - prior_weight`` far from it. Default 0.2, a tie-breaker.

    Returns
    -------
    ndarray (X, Y, Z)
    """
    s = {**SOFT_SCORES, **(scores or {})}
    cos_a = np.clip(np.abs(e1[..., FIBRE_AXIS[group]]), 0.0, 1.0)
    angle = np.degrees(np.arccos(cos_a))
    w = 1.0 - _rising(angle, *s["angle"])
    w = w * _rising(maps["fa"], *s["fa"])
    w = w * _rising(maps["cl"], *s["cl"])
    w = w * (1.0 - _rising(maps["cp"], *s["cp"]))
    w = w * (1.0 - _rising(maps["md"], *s["md"]))
    d2 = np.sum((coords - np.asarray(ALPS_MNI_COORDS[hemi][group])) ** 2, axis=-1)
    w = w * (1.0 - prior_weight * (1.0 - np.exp(-d2 / (2.0 * prior_sigma**2))))
    return np.nan_to_num(w)


_NEIGHBOURS = np.array(
    [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)], dtype=int
)


def _grow_region(seed, w, e1, allowed, budget, *, max_angle, min_weight):
    """Grow a 26-connected region from ``seed`` to ``budget`` voxels.

    At each step the highest-weight neighbour of the region joins it, if its
    weight is at least ``min_weight`` and its principal eigenvector is within
    ``max_angle`` of the region's mean principal eigenvector.
    """
    shape = np.array(w.shape)
    roi = np.zeros(w.shape, dtype=bool)
    roi[seed] = True
    mean_vec = e1[seed].astype(float).copy()
    cos_t = np.cos(np.radians(max_angle))
    frontier = set()

    def push(v):
        nb = np.asarray(v) + _NEIGHBOURS
        nb = nb[np.all((nb >= 0) & (nb < shape), axis=1)]
        for n in map(tuple, nb):
            if allowed[n] and not roi[n] and w[n] >= min_weight:
                frontier.add(n)

    push(seed)
    n = 1
    while n < budget and frontier:
        ref = mean_vec / np.linalg.norm(mean_vec)
        cands = [v for v in frontier if abs(e1[v] @ ref) >= cos_t]
        if not cands:
            break
        v = max(cands, key=lambda c: (w[c], c))
        frontier.discard(v)
        roi[v] = True
        n += 1
        e = e1[v]
        mean_vec += e if e @ mean_vec >= 0 else -e
        push(v)
    return roi


def find_alps_rois_v2(
    evecs,
    maps,
    coords,
    mask,
    *,
    anat=None,
    voxel_size=(2.0, 2.0, 2.0),
    radius=2.5,
    y_range=(-24.0, 0.0),
    z_range=(16.0, 32.0),
    x_min=18.0,
    x_span=30.0,
    ventricles=None,
    sphere=None,
    scores=None,
    prior_sigma=6.0,
    prior_weight=0.2,
    max_grow_angle=20.0,
    grow_min=0.1,
    min_weight=1.0,
):
    """Soft-weighted data-driven ALPS ROIs with region growing.

    For each hemisphere:

    1. The search box is the :func:`find_alps_rois` prior box.
    2. Each voxel gets a weight per fibre group (:func:`soft_weights`).
    3. The projection and association ROIs share a (y, z) centre: the one
       whose disc of radius ``radius`` maximises the smaller of the two group
       weight sums, with projection fibres medial to association fibres.
    4. Each ROI grows from the highest-weight voxel of the group in that disc
       (:func:`_grow_region`) to the voxel count of a sphere of radius
       ``radius``. Growth stays inside the box, within ``2 * radius`` of the
       centre in (y, z) and within ``2 * radius`` of the seed in 3D.
    5. If the best weight sum is below ``min_weight``, a group has no seed, or
       the grown projection ROI is not medial to the association ROI, both ROIs
       of the hemisphere fall back to the ``sphere`` ROIs with unit weights.

    The box is set in template coordinates; distances (disc, growth limit)
    and the voxel budget are in native mm, along template-aligned axes.

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
    anat : ndarray (X, Y, Z, 3), optional
        Native world coordinates (mm) rotated to template-aligned axes.
        Default: ``coords``.
    voxel_size : sequence of float
        Native voxel size in mm.
    radius : float, optional
        ROI size: the grower stops at ``round(4/3 pi radius^3 / voxel volume)``
        voxels. Default 2.5 mm (5 mm diameter sphere).
    ventricles : ndarray (X, Y, Z) of bool, optional
        Output of :func:`ventricle_mask`.
    sphere : dict, optional
        Fallback ROIs, ``{(hemisphere, group): bool mask}``
        (:func:`sphere_rois`). Without it, a fallback ROI is empty.
    scores, prior_sigma, prior_weight
        Passed to :func:`soft_weights`.
    max_grow_angle : float, optional
        Largest angle (deg) between a new voxel's principal eigenvector and
        the ROI mean.
    grow_min : float, optional
        Smallest weight of a voxel that may join an ROI.
    min_weight : float, optional
        Smallest weight sum per group in the chosen disc before fallback.

    Returns
    -------
    rois : dict
        ``{(hemisphere, group): bool mask}``.
    weights : dict
        ``{(hemisphere, group): float map}``, zero outside the ROI.
    info : dict
        Per hemisphere: box limits, centre, fallback flag and per-ROI QC.
    fields : dict
        ``{group: float map}``: the soft weights inside both search boxes.
    """
    anat = coords if anat is None else anat
    e1 = evecs[..., :, 0]
    X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]
    step = float(min(voxel_size))
    budget = max(1, int(round(4.0 / 3.0 * np.pi * radius**3 / np.prod(voxel_size))))
    groups = tuple(FIBRE_AXIS)

    rois, weights, info = {}, {}, {}
    fields = {g: np.zeros(mask.shape) for g in groups}
    for hemi, sign in HEMISPHERE_SIGN.items():
        box, x_lim, z_lim = _prior_box(
            sign * X,
            Y,
            Z,
            ventricles,
            x_min=x_min,
            x_span=x_span,
            y_range=y_range,
            z_range=z_range,
        )
        box &= mask
        w = {}
        for g in groups:
            wg = soft_weights(
                e1,
                maps,
                coords,
                g,
                hemi,
                scores=scores,
                prior_sigma=prior_sigma,
                prior_weight=prior_weight,
            )
            w[g] = np.where(box, wg, 0.0)
            fields[g] = np.maximum(fields[g], w[g])

        yz = np.column_stack([anat[..., 1][box], anat[..., 2][box]])
        xs = sign * anat[..., 0][box]
        wv = {g: w[g][box] for g in groups}

        best, best_score = None, -np.inf
        if yz.size:
            for y0 in np.arange(yz[:, 0].min(), yz[:, 0].max() + 1e-6, step):
                for z0 in np.arange(yz[:, 1].min(), yz[:, 1].max() + 1e-6, step):
                    disc = np.sum((yz - (y0, z0)) ** 2, axis=1) <= radius**2
                    sums = {g: float(wv[g][disc].sum()) for g in groups}
                    score = min(sums.values())
                    if score <= best_score or score <= 0:
                        continue
                    x_g = {g: np.average(xs[disc], weights=wv[g][disc]) for g in groups}
                    if x_g["proj"] >= x_g["assoc"]:
                        continue
                    best, best_score = (y0, z0), score

        hemi_info = {
            "x_range": list(x_lim),
            "z_range": list(z_lim),
            "centre_yz": None if best is None else [float(best[0]), float(best[1])],
            "best_min_weight_sum": float(max(best_score, 0.0)),
            "voxel_budget": budget,
            "fallback": True,
        }

        grown = {}
        if best is not None and best_score >= min_weight:
            d2 = (anat[..., 1] - best[0]) ** 2 + (anat[..., 2] - best[1]) ** 2
            disc = box & (d2 <= radius**2)
            cylinder = box & (d2 <= (2.0 * radius) ** 2)
            for g in groups:
                seeds = disc & (w[g] >= grow_min)
                if not seeds.any():
                    break
                flat = np.flatnonzero(seeds)
                seed = np.unravel_index(flat[np.argmax(w[g].ravel()[flat])], box.shape)
                # Keep the ROI compact: within 2 * radius of the seed in 3D.
                ball = np.sum((anat - anat[seed]) ** 2, axis=-1) <= (2.0 * radius) ** 2
                grown[g] = _grow_region(
                    seed,
                    w[g],
                    e1,
                    cylinder & ball,
                    budget,
                    max_angle=max_grow_angle,
                    min_weight=grow_min,
                )
            if len(grown) == len(groups):
                xs_g = {g: np.mean(sign * anat[..., 0][grown[g]]) for g in groups}
                if xs_g["proj"] >= xs_g["assoc"]:
                    grown = {}  # projection must stay medial to association

        if len(grown) == len(groups):
            hemi_info["fallback"] = False
            for g in groups:
                rois[(hemi, g)] = grown[g]
                weights[(hemi, g)] = np.where(grown[g], w[g], 0.0)
        else:
            logger.warning(
                "Data-driven ALPS ROI search failed in hemisphere %s; "
                "falling back to the sphere ROIs",
                hemi,
            )
            for g in groups:
                roi = (
                    sphere[(hemi, g)].copy()
                    if sphere is not None
                    else np.zeros(mask.shape, dtype=bool)
                )
                rois[(hemi, g)] = roi
                weights[(hemi, g)] = roi.astype(float)

        for g in groups:
            roi, wr = rois[(hemi, g)], weights[(hemi, g)][rois[(hemi, g)]]
            qc = {"nvox": int(roi.sum())}
            if qc["nvox"]:
                sw = float(wr.sum())
                centroid = np.average(coords[roi], axis=0, weights=wr + 1e-12)
                ref = np.asarray(ALPS_MNI_COORDS[hemi][g])
                qc.update(
                    sum_w=sw,
                    n_eff=float(sw**2 / np.sum(wr**2)) if sw > 0 else 0.0,
                    centroid_mni=[float(c) for c in centroid],
                    dist_to_sphere_mm=float(np.linalg.norm(centroid - ref)),
                )
                if sphere is not None:
                    overlap = (roi & sphere[(hemi, g)]).sum() / qc["nvox"]
                    qc["sphere_overlap"] = float(overlap)
            hemi_info[g] = qc
        info[hemi] = hemi_info
    return rois, weights, info, fields


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


ROI_METRIC_KEYS = (
    "Dxx",
    "Dyy",
    "Dzz",
    "nvox",
    "sum_w",
    "n_eff",
    "fa",
    "md",
    "cl",
    "cp",
    "angle_deg",
)


def roi_metrics(tensor, evecs, maps, roi, group, *, weights=None):
    """Mean diagonal diffusivities and quality metrics in one ROI.

    With ``weights``, every mean is a weighted mean, ``sum_w`` is the weight
    sum and ``n_eff`` the effective voxel count ``(sum w)^2 / sum w^2``.
    """
    n = int(roi.sum())
    out = dict.fromkeys(ROI_METRIC_KEYS, np.nan)
    out["nvox"] = n
    if n == 0:
        return out
    w = np.ones(n) if weights is None else np.asarray(weights)[roi]
    if not w.sum() > 0:
        logger.warning(
            "ROI weights of a %s ROI sum to %s; using unit weights", group, w.sum()
        )
        w = np.ones(n)
    out["sum_w"] = float(w.sum())
    out["n_eff"] = float(w.sum() ** 2 / np.sum(w**2))
    mean_tensor = np.tensordot(w, tensor[roi], axes=1) / w.sum()
    out["Dxx"], out["Dyy"], out["Dzz"] = np.diag(mean_tensor)
    for key in ("fa", "md", "cl", "cp"):
        out[key] = float(np.average(maps[key][roi], weights=w))
    cos_a = np.clip(np.abs(evecs[roi][:, FIBRE_AXIS[group], 0]), 0.0, 1.0)
    out["angle_deg"] = float(np.average(np.degrees(np.arccos(cos_a)), weights=w))
    return out


def _alps_ratio(proj, assoc):
    num = np.mean([proj["Dxx"], assoc["Dxx"]])
    den = np.mean([proj["Dyy"], assoc["Dzz"]])
    return float(num / den)


def alps_index(tensor, evecs, maps, rois, *, weights=None):
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
    weights : dict, optional
        ``{(hemisphere, group): float map}``. The index is then the ratio of
        weighted mean diffusivities; ``alps_unweighted`` uses unit weights in
        the same ROIs.

    Returns
    -------
    dict
        ``{hemisphere: {"alps", "alps_unweighted", "proj": {...},
        "assoc": {...}}}``, and ``"mean"`` and ``"mean_unweighted"``: the
        means of the hemispheric indices.
    """
    result = {}
    for hemi in HEMISPHERE_SIGN:
        m, u = {}, {}
        for g in ("proj", "assoc"):
            roi = rois[(hemi, g)]
            u[g] = roi_metrics(tensor, evecs, maps, roi, g)
            m[g] = (
                u[g]
                if weights is None
                else roi_metrics(
                    tensor, evecs, maps, roi, g, weights=weights[(hemi, g)]
                )
            )
        result[hemi] = {
            "alps": _alps_ratio(m["proj"], m["assoc"]),
            "alps_unweighted": _alps_ratio(u["proj"], u["assoc"]),
            "proj": m["proj"],
            "assoc": m["assoc"],
        }
    for key, src in (("mean", "alps"), ("mean_unweighted", "alps_unweighted")):
        result[key] = float(np.mean([result[h][src] for h in HEMISPHERE_SIGN]))
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
    roi_methods=ROI_METHODS,
    find_kwargs=None,
    legacy_kwargs=None,
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
    roi_methods : sequence of str
        Any of :data:`ROI_METHODS`.
    find_kwargs : dict, optional
        Keyword arguments of :func:`find_alps_rois_v2` (``data``).
    legacy_kwargs : dict, optional
        Keyword arguments of :func:`find_alps_rois` (``data-legacy``).

    Returns
    -------
    dict
        ``results[(method, frame)]`` from :func:`alps_index`; ``rois`` and
        ``weights`` keyed ``sphere`` or ``<method>-<frame>``; ``fields`` (soft
        weight maps of ``data``); ``frames`` (rotations) and ``info``.
    """
    unknown = set(roi_methods) - set(ROI_METHODS)
    if unknown:
        raise ValueError(f"Unknown ROI methods: {sorted(unknown)}")
    find_kwargs = dict(find_kwargs or {})
    legacy_kwargs = dict(legacy_kwargs or {})
    M = bvec_frame_to_world(affine, convention=bvec_convention)
    R = template_rotation(registration["final_affine"])
    Q = R.T @ M

    coords = registration.get("coords")
    if coords is None:
        coords = native_template_coords(registration, mask.shape, affine)
    valid = mask & np.all(np.isfinite(coords), axis=-1)

    # Native world mm on template-aligned axes: p -> R^T p for each voxel.
    affine = np.asarray(affine)
    ijk = np.indices(mask.shape).reshape(3, -1).T
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    anat = (world @ R).reshape(mask.shape + (3,))

    maps = eigen_maps(evals)
    voxel_size = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    ventricles = ventricle_mask(
        maps, coords, valid, voxel_volume=float(np.prod(voxel_size))
    )
    frames = {
        "bvec": (tensor, evecs),
        "template": (rotate_tensor(tensor, Q), rotate_vectors(evecs, Q)),
    }

    results, rois, weights, fields, info = {}, {}, {}, {}, {}
    sphere = sphere_rois(coords, valid)
    if "sphere" in roi_methods:
        rois["sphere"] = sphere
    for frame, (D, V) in frames.items():
        if "sphere" in roi_methods:
            results[("sphere", frame)] = alps_index(D, V, maps, sphere)
        if "data-legacy" in roi_methods:
            key = f"data-legacy-{frame}"
            rois[key], info[key] = find_alps_rois(
                V,
                maps,
                coords,
                valid,
                voxel_size=voxel_size,
                ventricles=ventricles,
                **legacy_kwargs,
            )
            results[("data-legacy", frame)] = alps_index(D, V, maps, rois[key])
        if "data" in roi_methods:
            key = f"data-{frame}"
            rois[key], weights[key], info[key], fields[key] = find_alps_rois_v2(
                V,
                maps,
                coords,
                valid,
                anat=anat,
                voxel_size=voxel_size,
                ventricles=ventricles,
                sphere=sphere,
                **find_kwargs,
            )
            results[("data", frame)] = alps_index(
                D, V, maps, rois[key], weights=weights[key]
            )

    return {
        "results": results,
        "rois": rois,
        "weights": weights,
        "fields": fields,
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

CSV_FIELDS = [
    "method",
    "frame",
    "hemisphere",
    "alps",
    "alps_mean",
    "alps_unweighted",
    "alps_mean_unweighted",
] + [f"{group}_{key}" for group in ("proj", "assoc") for key in ROI_METRIC_KEYS]


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
                "alps_unweighted": res[hemi]["alps_unweighted"],
                "alps_mean_unweighted": res["mean_unweighted"],
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


def roi_weight_image(weights):
    """4D float image, one volume per ROI in :data:`ROI_LABELS` order."""
    return np.stack([weights[key] for key in ROI_LABELS], axis=-1).astype(np.float32)


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
        k: out["rois"][k]
        for k in ("sphere", "data-legacy-template", "data-template")
        if k in out["rois"]
    }
    field = out["fields"].get("data-template")
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
    n = len(roi_sets) + 1 + (field is not None)
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

    if field is not None:
        ax = axes[len(roi_sets)]
        wmap = np.maximum(field["proj"], field["assoc"])[:, :, k].T
        im = ax.imshow(wmap, origin="lower", cmap="magma", vmin=0, vmax=1)
        for (_hemi, group), roi in out["rois"]["data-template"].items():
            sl = roi[:, :, k].T.astype(float)
            if sl.any():
                ax.contour(sl, levels=[0.5], colors=colours[group], linewidths=1)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("data soft weights (max of proj, assoc)")
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
    p.add_argument(
        "--roi-method",
        nargs="+",
        choices=ROI_METHODS,
        default=list(ROI_METHODS),
        help="ROI methods to run. Default: all.",
    )
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
    p.add_argument("--b0-threshold", type=float, default=B0_THRESHOLD)
    p.add_argument("--no-qc", action="store_true", help="Skip the QC figure.")
    return p


#: b-value (s/mm^2) at or below which a volume counts as b0.
B0_THRESHOLD = 50.0


def tensor_model(bvals, bvecs, *, b0_threshold=B0_THRESHOLD, **kwargs):
    """Gradient table and the WLS ``TensorModel`` used by the pipeline."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)
    return gtab, TensorModel(gtab, **kwargs)


def fit_and_register(
    data,
    affine,
    bvals,
    bvecs,
    *,
    mask=None,
    b0_threshold=B0_THRESHOLD,
    reg="affine",
    fit=True,
):
    """Brain mask, tensor fit and template registration of a DWI.

    Parameters
    ----------
    data : ndarray (X, Y, Z, N)
    affine : ndarray (4, 4)
    bvals, bvecs : ndarray
    mask : ndarray (X, Y, Z) of bool, optional
        Default: ``median_otsu`` on the b0 volumes.
    b0_threshold : float, optional
    reg : {"affine", "syn"}, optional
    fit : bool, optional
        Fit the tensor. With ``False`` the first output is ``None``.

    Returns
    -------
    fit : TensorFit or None
    mask : ndarray of bool
    registration : dict
        Output of :func:`register_to_template`.
    b0 : ndarray
        Mean b0 volume.
    """
    from dipy.segment.mask import median_otsu

    gtab, model = tensor_model(bvals, bvecs, b0_threshold=b0_threshold)
    b0_idx = np.flatnonzero(gtab.b0s_mask)
    b0 = data[..., b0_idx].mean(axis=-1)
    if mask is None:
        _, mask = median_otsu(
            data, vol_idx=b0_idx, median_radius=4, numpass=4, dilate=2
        )
        logger.info("Computed brain mask with median_otsu")

    tensor_fit = None
    if fit:
        logger.info("Fitting DTI on %d volumes", data.shape[-1])
        tensor_fit = model.fit(data, mask=mask)

    logger.info("Registering mean b0 to the MNI template (%s)", reg)
    registration = register_to_template(b0, affine, mask, method=reg)
    return tensor_fit, mask, registration, b0


def main(*, argv=None):
    import nibabel as nib

    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti, save_nifti

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, affine = load_nifti(args.dwi)
    bvals, bvecs = read_bvals_bvecs(args.bval, args.bvec)
    if args.max_bval is not None:
        keep = bvals <= args.max_bval
        data, bvals, bvecs = data[..., keep], bvals[keep], bvecs[keep]
    mask = nib.load(args.mask).get_fdata() > 0 if args.mask else None
    fit, mask, registration, b0 = fit_and_register(
        data,
        affine,
        bvals,
        bvecs,
        mask=mask,
        b0_threshold=args.b0_threshold,
        reg=args.reg,
    )
    if not args.mask:
        save_nifti(out_dir / "mask.nii.gz", mask.astype(np.uint8), affine)

    methods = tuple(m for m in ROI_METHODS if m in args.roi_method)
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
    for name, weights in out["weights"].items():
        save_nifti(
            out_dir / f"weights_{name}.nii.gz", roi_weight_image(weights), affine
        )
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
            "ALPS %-11s %-8s L=%.3f R=%.3f mean=%.3f (unweighted %.3f)",
            method,
            frame,
            res["L"]["alps"],
            res["R"]["alps"],
            res["mean"],
            res["mean_unweighted"],
        )
    logger.info(
        "Template rotation of the b-vector frame: %.2f deg", out["frames"]["tilt_deg"]
    )
    return out


if __name__ == "__main__":
    main()
