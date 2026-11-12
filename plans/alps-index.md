# Rotation-corrected, data-driven DTI-ALPS script for DIPY

## Context
Park et al. 2026 (npj Parkinson's Dis., doi 10.1038/s41531-026-01620-4) compute the DTI-ALPS index
with DIPY used only for the tensor fit; ROIs are 5 mm MNI spheres (proj L(-28,-12,24) R(24,-12,24),
assoc L(-40,-12,24) R(36,-12,24)) inverse-warped to native space, hand-corrected by a radiologist,
and diffusivities are read in the native (scanner) frame:

    ALPS = mean(Dxx_proj, Dxx_assoc) / mean(Dyy_proj, Dzz_assoc)

Goals (user decisions, 2026-11-10):
- Self-contained DIPY-based script, structured so it splits cleanly into `dipy/stats/alps.py` +
  `AlpsFlow` in `dipy/workflows/stats.py` later.
- Input: preprocessed DWI + bval/bvec + optional brain mask. If no mask is given, compute one with
  `dipy.segment.mask.median_otsu` on the b0 volumes (`vol_idx` = b0 indices) and save it. No other
  preprocessing in scope.
- Rotate tensors into a template-aligned frame before reading Dxx/Dyy/Dzz (head-orientation bias;
  same reasoning as bvec rotation in motion/eddy correction).
- New value: constrained data-driven ROI finder that rejects crossing/impure voxels; MNI spheres kept
  as the reference method.
- Outputs report the four diffusivities, FA, and purity metrics per ROI, not only the ratio; output is
  named "ALPS index", never "glymphatic function".
- Known unavoidable limits (diffusion scale vs PVS flow; acquisition dependence) are documented, not
  solved. Validation weakness and the multi-compartment idea feed the exploratory Phase 2.

## Location
- Branch `plan/alps` off `upstream/master` (not off `plan/tensor-na`).
- `alps/alps.py` (library functions + `argparse` `main`), `alps/test_alps.py` (pytest), `alps/README.md`.
  Not packaged; on upstreaming, functions move to `dipy/stats/alps.py` and `main` becomes `AlpsFlow`.
- Plan copy: `plans/alps-index.md` in the repo (no `plans/` dir yet; create on the branch).

## Reused DIPY pieces
- Fit: `dipy.reconst.dti.TensorModel` / `TensorFit.quadratic_form` (`dti.py:976`), `decompose_tensor`
  (`:2481`), `color_fa` (`:295`); IO `dipy.io.image.load_nifti`, `dipy.io.gradients.read_bvals_bvecs`.
- Registration: `dipy.align.affine_registration` (`align/_public.py:407`; `final_affine` maps template
  world -> native world), `register_dwi_to_template(..., reg_method="syn")` (`:181`) as option;
  `AffineMap.transform_inverse(..., interpolation="nearest")` (`imaffine.py:448`) for ROIs.
- Rotation extraction: `scipy.linalg.polar` (pattern from `reorient_bvecs`, `core/gradients.py:746`).
- Template: `dipy.data.read_mni_template(version="a", contrast="T2")` (register mean b0 to it).
- Purity/colour helpers: `segment_from_cfa` (`segment/mask.py:247`), `bounding_box` (`:73`).
- Workflow/test precedent to mirror: `SNRinCCFlow` (`workflows/stats.py:37`), `test_stats.py`.

## Phase 1 — core (`alps/alps.py`)
1. `fit_tensor(dwi, gtab, mask)` -> `TensorFit`.
2. `bvec_frame_to_world(affine, convention="fsl")` -> 3x3 orthogonal `M`: FSL bvecs are voxel-axis
   with x flipped when `det(affine[:3,:3]) > 0`; `M = polar(affine[:3,:3])[0] @ F`. Option
   `convention="world"` returns identity (bvecs already in RAS world).
3. `template_rotation(final_affine)` -> `R = polar(final_affine[:3,:3])[0]` (native world <- template).
4. `rotate_tensor(qform, Q)` -> `Q.T`-style einsum `D' = Q D Q^T` per voxel. Full frame:
   `Q = R.T @ M`, so `D_mni_frame = Q D_bvec Q^T`, kept on the native grid (no tensor resampling).
5. `mni_sphere_rois(coords, radius_mm=2.5, template, final_affine, native_shape, native_affine)` ->
   dict of 4 native-space boolean masks (spheres built on template grid, `transform_inverse` nearest).
6. `find_alps_rois(fit_rotated, mask, prior_box, ...)` — constrained data-driven finder:
   a. Prior box per hemisphere = MNI slab around the published coords (z 16–32 mm, y -24..0,
      |x| 18–48), mapped to native.
   b. Lateral-ventricle body: CSF-like voxels (MD > 2.0e-3, FA < 0.2) inside the brain mask; keep the
      two largest components near midline; slab = axial levels where ventricle body exists ∩ prior.
   c. Candidate voxels, from rotated eigenvectors: projection = |e1·z| ≥ cos 25°, association =
      |e1·y| ≥ cos 25°; both FA ≥ 0.3, linearity c_l = (λ1-λ2)/λ1 ≥ 0.4, planarity c_p low
      (rejects callosal crossings — the main ALPS confound).
   d. Geometry: per (y, z) row, require a projection run medial and an association run lateral,
      lateral to the ventricle wall (veins run along x). Score rows by purity; grow the ROI from the
      best rows until volume ≈ 5 mm sphere (≈65 mm³). All thresholds are keyword args.
7. `alps_index(qform_rot, rois)` -> per hemisphere: Dxx/Dyy/Dzz per ROI, ALPS L, R, mean; plus FA,
   MD, mean angular deviation, c_l, c_p, voxel count per ROI.
8. `main()`: args `dwi bval bvec [--mask PATH] --out-dir --roi-method {sphere,data,both} --bvec-convention
   {fsl,world} --no-rotation --reg {affine,syn}`. Writes `alps.csv` (one row per method × rotation
   on/off), ROI NIfTIs, `qc_alps.png` (colour FA in template frame, axial slice through ROIs, ROI
   outlines, ventricle outline).
9. `alps/README.md`: method, frame conventions, assumptions, and limitations list (scale, geometry,
   radial-diffusivity/crossing confound, non-specificity, orientation, acquisition, validation).

## Phase 2 — exploratory compartment analysis (after Phase 1 is audited)
Motivation: at b≈1000 a CSF-like PVS compartment (D≈3e-3 mm²/s) keeps only ~5% signal vs ~50% for
tissue, so standard ALPS mostly sees tissue radial diffusivity. Multi-shell data with low b allows:
- Per-shell ALPS (fit DTI per shell / b ≤ 500 vs b = 1000): b-dependence of the x-excess.
- FW-DTI (`dipy.reconst.fwdti.FreeWaterTensorModel`): ALPS on the tissue tensor vs on the full
  tensor, and FW fraction in each ROI — directly tests whether the x-excess lives in the fast pool.
- Note the user's MSMT-residual idea (subtract WM/GM predictions, read orientation of what is left)
  as related work; not implemented — residual orientation is dominated by WM response misfit.
Data: `read_cenir_multib` (b 200/400/1000/2000/3000). Scope decided after Phase 1.

## Verification
- Unit (`alps/test_alps.py`, synthetic): (1) `rotate_tensor` with identity / 90° permutations swaps
  diagonals as expected; (2) synthetic tensor field with known proj (z) / assoc (y) fibres plus x-excess
  → known ALPS; rotate the whole volume + affine by ±10° about each axis → rotation-corrected ALPS
  unchanged (≤1%), uncorrected drifts; (3) FSL-flip handling: same data with LAS vs RAS affine gives
  same result; (4) data-driven finder recovers planted ROIs and rejects planted crossing voxels.
- Real data: Sherbrooke 3-shell (b=1000 shell) and CFIN; compare sphere vs data-driven ROIs
  (centroid distance, ALPS agreement), rotation on vs off; inspect `qc_alps.png`. Expect ALPS ≈1.1–1.7
  (paper cohort 1.24 ± 0.22).
- [x] Robustness demo: resample a real DWI with a synthetic 5–10° head tilt (rotate image and bvecs) and
  show corrected vs uncorrected ALPS (`experiments.py tilt`).
- `ruff` clean; pytest passes. Show results to user; no commit until user audits.

## As built (2026-11-12)
- Files: `alps/alps.py` (method), `alps/test_alps.py` (16 synthetic tests),
  `alps/datasets.py` (local datasets: `sherbrooke`, `hbn`, `stanford`),
  `alps/experiments.py` (`direct`, `tilt`, `regdiag`, `summary` subcommands; replaces the
  earlier `tilt_demo.py` / `reg_diag.py`), `alps/README.md`, `alps/RESULTS.md` (findings).
  Results in `alps_results/` (git-excluded via `.git/info/exclude`).
- Registration: skull-stripped mean b0 (b0 × brain mask) → skull-stripped MNI 2009a T2
  (× MNI 2009c brain mask), resampled to 2 mm; affine, MI, dense sampling,
  `level_iters=(200, 50, 10)`; about 1 min per registration. Note: DIPY silently drops
  `static_mask`/`moving_mask` when `sampling_proportion < 1`, so masks are applied by
  multiplying the images.
- Validation datasets are the local clinical-scale ones (~2 mm, 1–2 shells). Group tests
  across subjects are deferred until the feature set is complete and a comparison plan is
  written.
- Open: data-driven ROI finder (hard thresholds) gives NaN or large L/R differences under
  small perturbations → rework next (soft-weighted ROIs, fixed-volume region growing;
  RecoBundles CST/SLF as an anatomical reference).
