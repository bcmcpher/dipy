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
Original design (2026-11-10). Where the build differs, "As built" below is authoritative.
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
   d. Geometry: projection candidates medial to association candidates, lateral to the
      ventricle wall (veins run along x); the ROI centre is the shared (y, z) disc with the most
      voxels of the smaller group. All thresholds are keyword args. (Superseded by v2, Phase 1b.)
7. `alps_index(qform_rot, rois)` -> per hemisphere: Dxx/Dyy/Dzz per ROI, ALPS L, R, mean; plus FA,
   MD, mean angular deviation, c_l, c_p, voxel count per ROI.
8. `main()`: args `dwi bval bvec [--mask PATH] --out-dir --roi-method --bvec-convention
   {fsl,voxel,world} --reg {affine,syn} --max-bval`. Writes `alps.csv` (one row per method × frame
   (`bvec`, `template`) × hemisphere), ROI label NIfTIs (`rois_<method>.nii.gz`), `qc_alps.png` (colour FA in template frame, axial slice through ROIs, ROI
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
Data: the local 2-shell datasets first (`hbn` b 1000/2000, `sherbrooke` b 1000/2000/3000);
FW-DTI needs ≥ 2 non-zero shells, so single-shell clinical data (and `stanford`) are out of
scope for it. `read_cenir_multib` (b 200–3000) only for the per-shell low-b analysis.
Scope decided after Phase 1.

## Verification
- Unit (`alps/test_alps.py`, synthetic): (1) `rotate_tensor` with identity / 90° permutations swaps
  diagonals as expected; (2) synthetic tensor field with known proj (z) / assoc (y) fibres plus x-excess
  → known ALPS; rotate the whole volume + affine by ±10° about each axis → rotation-corrected ALPS
  unchanged (≤1%), uncorrected drifts; (3) FSL-flip handling: same data with LAS vs RAS affine gives
  same result; (4) data-driven finder recovers planted ROIs and rejects planted crossing voxels.
- [x] Real data: Sherbrooke (b=1000 shell), HBN (b=1000), Stanford (b=2000); sphere vs
  data-driven ROIs, `bvec` vs `template` frame; inspect `qc_alps.png`. (CFIN dropped: thin slab.)
  Paper cohort 1.24 ± 0.22; local values 1.3–1.8.
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
  small perturbations → rework in Phase 1b.
- Known limit: the tuning subject (HBN) is paediatric. FA levels and white-matter lesions
  differ in older brains, so default weight parameters may not transfer; test in the group
  phase and state it in the README limitations.

## Phase 1b — data-driven ROI v2 (decided 2026-11-12)
Problem: the v1 finder (`find_alps_rois`, hard thresholds on FA, c_l, c_p and a 25° angle
cone) returns NaN in 2/12 HBN tilt runs and L/R differences up to 22%. One voxel crossing a
threshold changes a 3–7 voxel ROI.

Decisions:
- Tensor-only. No CSD/fODF or other non-tensor model in the screen.
- ROI size is set by a radius (`radius_mm`, default 2.5 mm, i.e. the paper's 5 mm diameter
  sphere). The grower stops at the voxel count of a sphere of that radius on the native
  grid: `round(4/3 π r³ / voxel_volume)`.
- Primary estimate is the weighted ALPS; unweighted ALPS is also reported.
- The v1 finder stays as `data-legacy`, so the improvement can be reported against it.
  `--roi-method` takes a list: `sphere data data-legacy` (default all three).

Method (`find_alps_rois_v2`):
1. Search region: the v1 per-hemisphere prior box at the lateral-ventricle body level,
   lateral to the ventricle wall. Template coordinates only gate the box; distances and
   volumes are in native mm (the affine scale is 0.73–0.91 on the test data).
2. Soft voxel weight in [0, 1], the product of:
   - orientation: smooth fall-off of |e1 · axis| (z for projection, y for association);
   - linearity c_l: rising sigmoid;
   - planarity c_p: falling sigmoid (crossing-fibre penalty);
   - FA: rising sigmoid;
   - MD: falling sigmoid (partial volume with CSF next to the ventricle);
   - weak Gaussian prior around the published MNI centre (σ ≈ 6 mm) for tie-breaks.
   Each sigmoid has a centre and a width as keyword arguments; v1 thresholds are the
   default centres.
3. Pair constraint: projection and association ROIs share a (y, z) centre (the medullary
   veins run along x through both). Pick the centre that maximises
   min(Σw_proj, Σw_assoc) over a disc of radius `radius_mm`; projection medial to
   association.
4. Region growing per group: seed at the highest-weight voxel near the centre; add the
   highest-weight 26-connected neighbour whose e1 is within ~20° of the ROI mean e1; stop at
   the voxel budget from `radius_mm`.
5. ALPS: ratio of weighted means, not a mean of per-voxel ratios. For each ROI,
   `D̄ = Σ wᵢ Dᵢ / Σ wᵢ` for Dxx, Dyy and Dzz; then
   `ALPS = mean(D̄xx_proj, D̄xx_assoc) / mean(D̄yy_proj, D̄zz_assoc)`. The unweighted
   version (all wᵢ = 1 inside the grown ROI) is the second result.
6. Failure: if the best min(Σw) is below a floor, fall back to the sphere ROI and set
   `fallback=True`. No silent NaN.
7. Per-ROI QC: Σw, effective voxel count (Σw)²/Σw², mean angle to the axis, c_l, c_p,
   distance to the sphere centre, overlap with the sphere ROI, `fallback`. Weight map in
   `qc_alps.png`.
8. ROI export for review: every method writes its final ROIs as `rois_<method>.nii.gz`
   (labels 1–4, as now) and, for weighted methods, `weights_<method>.nii.gz` (float, one
   volume per ROI, zero outside the grown ROI), both on the native grid. Lets a reviewer
   check placement in any viewer, as the paper's radiologist did.

Evaluation (local datasets, `experiments.py`). HBN is the tuning set; Sherbrooke and
Stanford are held out: run with the HBN-tuned defaults, reported, never re-tuned.
- New `sensitivity` subcommand: perturb each soft-score centre and width by ±10% and ±20%,
  rerun the ROI step only (fit and registration cached), report the ALPS change.
- New `bootstrap` subcommand (measurement noise): residual (wild) bootstrap of the DWI
  around the tensor fit, N ≈ 50–100 replicates, ROI step and ALPS per replicate. Reports
  the SD and CV of ALPS per method. This is the noise floor against which the tilt CV and
  the sensitivity are judged.
- New `planted` subcommand (validity): synthesise a DWI from a subject's fitted tensors
  with a known extra Dxx (e.g. +5%, +10%, +20%) in the periventricular white matter of the
  ALPS slab, add Rician noise at the measured SNR, and check that each ROI method recovers
  the known ALPS change (bias and slope). Also plant a crossing-fibre perturbation to check
  that v2 rejects it.
- New `montage` subcommand: one image per dataset with the ROIs of every method and run
  (tilt angles, bootstrap examples) on colour FA, for a quick visual audit.
- Registration: run the sphere and `data` methods with `--reg affine` and `--reg syn` on
  every dataset; report the change in ROI centre and ALPS.
- Pass criteria on HBN: no NaN and no fallback in the tilt series and the ±0.5° runs;
  template-frame CV across tilts ≤ the sphere method (1.7%) and ≤ the bootstrap CV;
  L/R difference stable under the tilt series and the perturbations (its spread no larger
  than the sphere method's; no absolute symmetry target, because real asymmetry exists);
  ALPS change ≤ 3% for ±10% perturbations; planted-effect slope within 10% of 1.
- `RESULTS.md` compares `sphere`, `data-legacy` and `data` on every metric.
- Unit tests: phantom recovery (planted bundles, crossing rejection, as for v1); grown ROI
  size equals the voxel budget; fallback triggers on an isotropic phantom; weights are
  continuous under a small rotation of the phantom.

As built (2026-11-13):
- [x] `find_alps_rois_v2` (method `data`), v1 kept as `data-legacy`, `--roi-method` list,
  weighted and unweighted ALPS, `weights_<method>.nii.gz`, soft-weight QC panel. Method
  steps 1–8 as above. Open details fixed in code: growth stays within `2 * radius` of the
  pair centre in (y, z); the MNI prior factor falls to `1 - prior_weight` (0.2); a voxel
  joins at weight ≥ `grow_min` (0.1); fallback below a weight sum of `min_weight` (1.0).
  Distances use native world mm rotated by `R^T` (template-aligned axes).
- [x] Unit tests: phantom recovery, crossing rejection, ROI size equals the voxel budget,
  fallback on an isotropic phantom, weight continuity under a 1° rotation.
- [x] `sensitivity` subcommand (soft-score centres and widths, ±10% and ±20%; ROI Dice
  and fallbacks). Results in `RESULTS.md`: largest change at ±10% is 3.3% on HBN,
  3.5% Sherbrooke, 3.0% Stanford; all at or just above the ≤ 3% criterion.
- [x] Growth bounded to 2 × `radius` from the seed in 3D and medial order checked on
  the grown ROIs (code review).
- [x] HBN tilt series with v2: no NaN or fallback, template CV 0.6%, L/R SD 2.3 points.
- [ ] `bootstrap`, `planted`, `montage`, affine vs SyN; Sherbrooke tilt rerun with v2.

Not planned (revisit only with a concrete need):
- Subject-level bundle segmentation (whole-brain tracking + RecoBundles): replaced by the
  atlas tract prior in Phase 1c, which needs no subject tractography.
- fODF peak count as a crossing screen: excluded by the tensor-only decision.
- Vascular atlases: deep medullary veins (~0.1–0.5 mm, variable) are not resolved by
  population venous atlases at DWI resolution; no localising value beyond the ALPS
  geometry assumption.

## Phase 1c — atlas tract prior (after Phase 1b passes its criteria)
Goal: replace the hand-set prior box and the Gaussian centre prior of v2 with tract
probability maps, so the search stays in the target bundles (projection: corona
radiata/CST; association: SLF/arcuate). Tensor scoring is unchanged.

1. Build MNI tract probability maps once from the DIPY HCP842 bundle atlas
   (`fetch_bundle_atlas_hcp842`): streamline density per voxel for CST, SLF and AF,
   normalised to [0, 1], on the 2 mm template grid. Cache them next to the template.
   - Check first: the atlas template space against MNI 2009a, and that both bundles have
     enough density at the lateral-ventricle body level.
   - If the spaces differ, measure the offset once (register the atlas template to 2009a)
     and apply it to the maps before caching.
2. Map the probability maps to native space with the existing registration
   (`native_template_coords` sampling).
3. v2 weight = tensor purity × orientation × tract probability (projection map for the
   projection group, association map for the association group). The box remains as a
   fallback when a tract map is empty in the slab.
4. New ROI method `data-tract`; report against `sphere`, `data-legacy` and `data` with the
   Phase 1b metrics and pass criteria.
5. Licensing: HCP842 ships with DIPY, so `data-tract` works without external files.

## Phase 1d — external comparison: JHU-label method (decided 2026-11-12)
Goal: replicate a published automated ALPS method as an external reference for the
improvement claim. It uses the JHU ICBM-DTI-81 white-matter labels: superior corona radiata
(projection) and superior longitudinal fasciculus (association), at the lateral-ventricle
body level (Zhang et al. 2021, cited in the README).

1. Before implementing, take the exact procedure from the paper: label numbers, slice or
   slab selection, ROI erosion or size, and the registration target and type (probably
   nonlinear, FA to an FA template). Replicate the registration where DIPY allows (SyN);
   list every remaining difference in `RESULTS.md`, because each one loosens the comparison.
2. The atlas is not distributed with this tool. The user passes its path, e.g.
   `--jhu-labels $FSLDIR/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz`. ROI method `jhu` is
   available only when the path is given.
3. Space: the JHU labels are in FSL MNI152 space, not MNI 2009a. Measure the offset once
   (register the FSL MNI152 T1 to 2009a) and either apply it or document it.
4. Report `jhu` against `sphere`, `data-legacy`, `data` and `data-tract` with the Phase 1b
   metrics. It is a comparison, not gated by pass criteria.
5. Unit tests use a synthetic label image; no FSL file in the test suite.
