"""Synthetic tests for alps.py. Run with ``pytest alps/test_alps.py``."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial.transform import Rotation

import alps as A
from dipy.reconst.dti import decompose_tensor

TRUE_ALPS = (0.5 + 0.5) / (0.3 + 0.3)


def _rot(angles_deg):
    return Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()


def phantom(*, crossing=False):
    """Template-space phantom on a 2 mm grid with ALPS geometry.

    - Lateral ventricles (isotropic 3e-3) at |x| < 16.
    - Projection band (principal direction z) at 20 <= |x| <= 30.
    - Association band (principal direction y) at 34 <= |x| <= 46.
    - Isotropic 0.7e-3 elsewhere.

    With ``crossing=True`` planar (crossing-fibre) tensors with principal
    direction z are planted in the projection band near the ROI centre.
    """
    xs = np.arange(-60, 61, 2.0)
    ys = np.arange(-50, 31, 2.0)
    zs = np.arange(0, 45, 2.0)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack([X, Y, Z], axis=-1)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [xs[0], ys[0], zs[0]]

    D = np.zeros(X.shape + (3, 3))
    D[...] = np.eye(3) * 0.7e-3
    ax = np.abs(X)
    ybox = (Y >= -30) & (Y <= 6)
    vent = (ax < 16) & (Y >= -40) & (Y <= 20) & (Z >= 4) & (Z <= 40)
    proj = (ax >= 20) & (ax <= 30) & ybox & (Z >= 12) & (Z <= 36)
    assoc = (ax >= 34) & (ax <= 46) & ybox & (Z >= 12) & (Z <= 36)
    D[vent] = np.eye(3) * 3.0e-3
    D[proj] = np.diag([0.5, 0.3, 1.7]) * 1e-3
    D[assoc] = np.diag([0.5, 1.7, 0.3]) * 1e-3
    cross = np.zeros(X.shape, dtype=bool)
    if crossing:
        cross = proj & (Y >= -14) & (Y <= -10) & (Z >= 22) & (Z <= 26)
        D[cross] = np.diag([1.0, 0.3, 1.2]) * 1e-3
    mask = np.ones(X.shape, dtype=bool)
    return D, coords, affine, mask, cross


def run(D, coords, affine, mask, *, final_linear=None, convention="world"):
    evals, evecs = decompose_tensor(D)
    final_affine = np.eye(4)
    if final_linear is not None:
        final_affine[:3, :3] = final_linear
    registration = {"final_affine": final_affine, "coords": coords}
    return A.compute_alps(
        D, evecs, evals, mask, affine, registration, bvec_convention=convention
    )


def test_rotate_tensor_identity_and_permutation():
    D = np.diag([1.0, 2.0, 3.0])
    npt.assert_allclose(A.rotate_tensor(D, np.eye(3)), D)
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    npt.assert_allclose(np.diag(A.rotate_tensor(D, P)), [3.0, 1.0, 2.0])


def test_rotate_tensor_roundtrip():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(4, 3, 3))
    D = B @ np.swapaxes(B, -1, -2)
    R = _rot([10, -20, 35])
    npt.assert_allclose(A.rotate_tensor(A.rotate_tensor(D, R), R.T), D, atol=1e-12)


def test_bvec_frame_fsl_flip_consistent_between_ras_and_las():
    ras = np.diag([2.0, 2.0, 2.0, 1.0])
    las = np.diag([-2.0, 2.0, 2.0, 1.0])
    M_ras = A.bvec_frame_to_world(ras, convention="fsl")
    M_las = A.bvec_frame_to_world(las, convention="fsl")
    # FSL bvec (1, 0, 0) points to world left (-x) for both storage orders.
    npt.assert_allclose(M_ras, np.diag([-1.0, 1.0, 1.0]))
    npt.assert_allclose(M_las, M_ras)
    npt.assert_allclose(A.bvec_frame_to_world(ras, convention="voxel"), np.eye(3))
    npt.assert_allclose(A.bvec_frame_to_world(ras, convention="world"), np.eye(3))


def test_template_rotation_ignores_scaling():
    R = _rot([5, 0, -12])
    final = np.eye(4)
    final[:3, :3] = R @ np.diag([1.1, 0.9, 1.05])
    npt.assert_allclose(A.template_rotation(final), R, atol=0.05)
    pure = np.eye(4)
    pure[:3, :3] = R
    npt.assert_allclose(A.template_rotation(pure), R, atol=1e-12)


def test_frame_tilt_ignores_flips():
    Q = np.diag([-1.0, 1.0, 1.0]) @ _rot([0, 0, 0])
    total, per_axis = A.frame_tilt(Q)
    npt.assert_allclose(total, 0.0, atol=1e-6)
    npt.assert_allclose(per_axis, 0.0, atol=1e-6)
    total, _ = A.frame_tilt(_rot([10, 0, 0]) @ np.diag([-1.0, 1.0, 1.0]))
    npt.assert_allclose(total, 10.0, atol=1e-6)


@pytest.mark.parametrize("method", A.ROI_METHODS)
def test_phantom_alps(method):
    D, coords, affine, mask, _ = phantom()
    out = run(D, coords, affine, mask)
    for frame in ("bvec", "template"):
        res = out["results"][(method, frame)]
        for hemi in ("L", "R"):
            assert res[hemi]["proj"]["nvox"] > 0
            assert res[hemi]["assoc"]["nvox"] > 0
            npt.assert_allclose(res[hemi]["alps"], TRUE_ALPS, rtol=1e-6)
            npt.assert_allclose(res[hemi]["alps_unweighted"], TRUE_ALPS, rtol=1e-6)


@pytest.mark.parametrize("angles", [[10, 0, 0], [0, 10, 0], [0, 0, 10], [-8, 6, 5]])
def test_rotation_correction_recovers_alps(angles):
    D, coords, affine, mask, _ = phantom()
    R = _rot(angles)
    D_native = A.rotate_tensor(D, R)  # tilted head: tensors in the native frame
    out = run(D_native, coords, affine, mask, final_linear=R)
    corrected = out["results"][("sphere", "template")]["mean"]
    uncorrected = out["results"][("sphere", "bvec")]["mean"]
    npt.assert_allclose(corrected, TRUE_ALPS, rtol=1e-6)
    assert abs(uncorrected - TRUE_ALPS) / TRUE_ALPS > 0.01
    npt.assert_allclose(out["frames"]["tilt_deg"], A.rotation_angle(R), atol=1e-6)
    for method in ("data", "data-legacy"):
        data = out["results"][(method, "template")]["mean"]
        npt.assert_allclose(data, TRUE_ALPS, rtol=1e-6)


def test_fsl_convention_end_to_end():
    """FSL bvecs on a RAS image: x of the fitted tensor is flipped."""
    D, coords, affine, mask, _ = phantom()
    F = np.diag([-1.0, 1.0, 1.0])
    D_fsl = A.rotate_tensor(D, F)
    out = run(D_fsl, coords, affine, mask, convention="fsl")
    npt.assert_allclose(out["results"][("sphere", "template")]["mean"], TRUE_ALPS)
    npt.assert_allclose(out["frames"]["tilt_deg"], 0.0, atol=1e-6)


@pytest.mark.parametrize("method", ["data", "data-legacy"])
def test_data_driven_rejects_crossing_voxels(method):
    D, coords, affine, mask, cross = phantom(crossing=True)
    out = run(D, coords, affine, mask)
    rois = out["rois"][f"{method}-template"]
    for hemi in ("L", "R"):
        roi = rois[(hemi, "proj")]
        assert roi.any()
        assert not (roi & cross).any()
    # The sphere ROIs do include them, so their index is biased.
    assert (out["rois"]["sphere"][("R", "proj")] & cross).any() or (
        out["rois"]["sphere"][("L", "proj")] & cross
    ).any()
    npt.assert_allclose(
        out["results"][(method, "template")]["mean"], TRUE_ALPS, rtol=1e-6
    )


@pytest.mark.parametrize("method", ["data", "data-legacy"])
def test_data_driven_proj_medial_to_assoc(method):
    D, coords, affine, mask, _ = phantom()
    out = run(D, coords, affine, mask)
    rois = out["rois"][f"{method}-template"]
    for hemi, sign in A.HEMISPHERE_SIGN.items():
        x_proj = sign * coords[rois[(hemi, "proj")]][:, 0]
        x_assoc = sign * coords[rois[(hemi, "assoc")]][:, 0]
        assert x_proj.max() < x_assoc.min()


@pytest.mark.parametrize("radius", [2.5, 4.0])
def test_v2_roi_size_is_voxel_budget(radius):
    D, coords, affine, mask, _ = phantom()
    evals, evecs = decompose_tensor(D)
    registration = {"final_affine": np.eye(4), "coords": coords}
    out = A.compute_alps(
        D,
        evecs,
        evals,
        mask,
        affine,
        registration,
        bvec_convention="world",
        roi_methods=("data",),
        find_kwargs={"radius": radius},
    )
    budget = int(round(4 / 3 * np.pi * radius**3 / 8.0))
    for hemi, hi in out["info"]["data-template"].items():
        assert not hi["fallback"]
        assert hi["voxel_budget"] == budget
        for group in ("proj", "assoc"):
            roi = out["rois"]["data-template"][(hemi, group)]
            assert roi.sum() == budget
            # Compact: every voxel within 2 * radius of the seed, so no two
            # voxels are more than 4 * radius apart.
            pts = coords[roi]
            span = np.linalg.norm(pts[:, None] - pts[None], axis=-1).max()
            assert span <= 4 * radius + 1e-6
            w = out["weights"]["data-template"][(hemi, group)]
            assert np.all(w[~roi] == 0)
            assert np.all((w[roi] > 0) & (w[roi] <= 1))


def test_v2_falls_back_on_isotropic_phantom():
    D, coords, affine, mask, _ = phantom()
    D[...] = np.eye(3) * 0.7e-3
    out = run(D, coords, affine, mask)
    for frame in ("bvec", "template"):
        for hi in out["info"][f"data-{frame}"].values():
            assert hi["fallback"]
        rois = out["rois"][f"data-{frame}"]
        for key, roi in rois.items():
            npt.assert_array_equal(roi, out["rois"]["sphere"][key])
        res = out["results"][("data", frame)]
        npt.assert_allclose(res["mean"], 1.0, rtol=1e-6)
        assert np.isfinite(res["mean_unweighted"])


def test_v2_weights_continuous_under_small_rotation():
    D, coords, affine, mask, _ = phantom()
    base = run(D, coords, affine, mask)
    # Uncorrected 1 deg tilt of the tensors: the bvec frame sees it.
    tilted = run(A.rotate_tensor(D, _rot([1, 0, 0])), coords, affine, mask)
    for key, roi in base["rois"]["data-bvec"].items():
        npt.assert_array_equal(tilted["rois"]["data-bvec"][key], roi)
        w0 = base["weights"]["data-bvec"][key]
        w1 = tilted["weights"]["data-bvec"][key]
        assert np.max(np.abs(w1 - w0)) < 0.02
    for field in ("proj", "assoc"):
        f0 = base["fields"]["data-bvec"][field]
        f1 = tilted["fields"]["data-bvec"][field]
        assert np.max(np.abs(f1 - f0)) < 0.02


def test_results_rows_and_labels(tmp_path):
    D, coords, affine, mask, _ = phantom()
    out = run(D, coords, affine, mask)
    rows = A.results_rows(out["results"])
    assert len(rows) == 3 * 2 * 2  # methods x frames x hemispheres
    A.write_csv(rows, tmp_path / "alps.csv")
    labels = A.roi_label_image(out["rois"]["sphere"])
    assert set(np.unique(labels)) == {0, 1, 2, 3, 4}
    weights = A.roi_weight_image(out["weights"]["data-template"])
    assert weights.shape == mask.shape + (4,)
    for i, key in enumerate(A.ROI_LABELS):
        npt.assert_array_equal(weights[..., i] > 0, out["rois"]["data-template"][key])


def test_tilt_dwi_moves_anatomy_and_bvecs_together():
    from experiments import tilt_dwi, tilt_matrix

    # 90 deg about x on an odd grid maps voxel centres onto voxel centres.
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-8.0, -8.0, -8.0]
    data = np.zeros((9, 9, 9, 1))
    data[4, 6, 4, 0] = 1.0  # 4 mm anterior of the centre
    bvecs = np.array([[0.0, 1.0, 0.0]])  # along y (FSL: x is flipped, y is not)
    Rt = tilt_matrix(90.0, "x")

    tilted, new_bvecs = tilt_dwi(data, affine, bvecs, Rt)
    # Anterior (+y) goes to superior (+z) under +90 deg about x.
    assert np.unravel_index(np.argmax(tilted[..., 0]), (9, 9, 9)) == (4, 4, 6)
    M = A.bvec_frame_to_world(affine, convention="fsl")
    npt.assert_allclose(M @ new_bvecs[0], Rt @ (M @ bvecs[0]), atol=1e-12)
    npt.assert_allclose(M @ new_bvecs[0], [0, 0, 1], atol=1e-12)


def test_wild_bootstrap_keeps_residual_magnitude():
    from experiments import wild_bootstrap_signal

    rng = np.random.default_rng(1)
    predicted = rng.uniform(100, 200, size=(500, 1, 1, 30))
    signal = predicted + rng.normal(0, 5, size=predicted.shape)
    boot = wild_bootstrap_signal(signal, predicted, rng)
    npt.assert_allclose(np.abs(boot - predicted), np.abs(signal - predicted))
    # Signs are random, so the replicate is unbiased around the prediction.
    flips = np.sign(boot - predicted) != np.sign(signal - predicted)
    assert 0.45 < flips.mean() < 0.55
