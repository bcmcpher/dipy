# Rotation-corrected DTI-ALPS

`alps.py` computes the DTI-ALPS index from preprocessed diffusion MRI with DIPY. DTI-ALPS
means diffusion tensor image analysis along the perivascular space. The script is
self-contained, and its functions are written to move into `dipy/stats/alps.py` later.

## Method

The ALPS index compares two diffusivities next to the body of the lateral ventricle:

- diffusivity along x (left-right), the assumed direction of the medullary veins;
- diffusivity perpendicular to both x and the local fibre direction.

```
ALPS = mean(Dxx_proj, Dxx_assoc) / mean(Dyy_proj, Dzz_assoc)
```

`proj` is a projection-fibre ROI (fibres along z). `assoc` is an association-fibre ROI
(fibres along y). The script computes the index for each hemisphere and their mean.

## What this script adds

1. **Rotation into the template frame.** The diagonal tensor elements depend on the frame.
   A fitted tensor is in the b-vector frame, which follows the scanner, not the head.
   The script registers the skull-stripped mean b0 (b0 × brain mask) to the
   skull-stripped MNI 2009a T2 template at 2 mm (affine, mutual information).
   The brain mask of the template is the MNI 2009c mask.
   The script then rotates each tensor with
   `D' = Q D Q^T`. `Q` combines the b-vector-to-world rotation and the rotation part of
   the registration. The tensors stay on the native grid, so no tensor interpolation occurs.
2. **Both frames in the output.** Each result is reported in the `bvec` frame (as in most
   published pipelines) and in the `template` frame. The difference shows the effect of
   head orientation.
3. **Data-driven ROIs.** The search finds coherent fibres lateral to the ventricle body.
   Voxels with low linearity or high planarity (crossing fibres) get low weights. The
   published 5 mm spheres are kept as the reference method.
4. **Quality metrics.** Each ROI reports its voxel count, weight sum, effective voxel
   count, FA, MD, linearity, planarity and the mean angle between the principal
   eigenvector and the expected fibre axis.

ROI methods:

| Method | ROIs | Index |
|---|---|---|
| `sphere` | 5 mm spheres at the published MNI coordinates | Mean of the voxels |
| `data` | Soft-weighted search with region growing (see below) | Ratio of weighted means |
| `data-legacy` | First search, with hard thresholds | Mean of the voxels |

A **soft weight** is a per-voxel score in [0, 1] that the `data` method uses in place of
hard thresholds. For `data`, `alps` is the weighted index. `alps_unweighted` is the index
in the same ROIs with unit weights. For the other methods the two columns are equal.

## Usage

```bash
python alps/alps.py dwi.nii.gz dwi.bval dwi.bvec --mask mask.nii.gz --out-dir out
```

Options:

- `--mask`: brain mask. If you omit it, the script computes one with `median_otsu` on the b0
  volumes and saves it as `mask.nii.gz`.
- `--max-bval 1100`: keep only b0 and the b=1000 shell of multi-shell data.
- `--bvec-convention {fsl,voxel,world}`: frame of the b-vectors. The default `fsl` is the
  FSL and dcm2niix convention: voxel axes, with x flipped when the affine determinant is
  positive.
- `--roi-method sphere data-legacy data`: one or more ROI methods. Default: all three.
- `--reg {affine,syn}`: registration used to map ROIs. The rotation always comes from the
  affine step.

Outputs in `--out-dir`:

| File | Content |
|---|---|
| `alps.csv` | One row per method, frame and hemisphere: ALPS, the diagonal diffusivities and the ROI metrics |
| `alps.json` | Rotations (`M`, `R`, `Q`), head tilt, registration affine, data-driven search details |
| `rois_<method>.nii.gz` | ROI labels: 1 L proj, 2 L assoc, 3 R proj, 4 R assoc |
| `weights_<method>.nii.gz` | `data` only: 4D soft weights, one volume per ROI in label order, zero outside the ROI |
| `qc_alps.png` | ROIs of each method on colour FA (template frame), ventricle outline, soft weights, registration check |
| `mask.nii.gz` | Brain mask, if the script computed it |

Check `qc_alps.png` for each subject. Registration errors move the sphere ROIs.

## Experiments

`alps/experiments.py` runs the validation experiments on the local datasets in
`alps/datasets.py` (`sherbrooke`, `hbn`, `stanford`). Results go to
`alps_results/<dataset>/<experiment>/`, which git ignores. Run it from the repo root.

```bash
python alps/experiments.py direct --dataset hbn
python alps/experiments.py tilt --dataset hbn --angles 0 0.5 5 10 -5 -10
python alps/experiments.py regdiag --dataset hbn
python alps/experiments.py sensitivity --dataset hbn
python alps/experiments.py bootstrap --dataset hbn --replicates 50
python alps/experiments.py summary
```

- `direct` runs `alps.py` on the dataset.
- `tilt` rotates the anatomy and the b-vectors about the image centre, which emulates a
  tilted head, and reruns the full pipeline at each angle. It deletes the tilted DWIs
  unless you pass `--keep-dwi`.
- `regdiag` registers a tilted mean b0 to the template. It reports the rotation error and
  the ROI placement error against the known tilt.
- `sensitivity` moves each soft-score centre and width of the `data` method by ±10% and
  ±20% (`--factors`). It fits and registers once, then reruns only the ROI step. It reports
  the ALPS change, the ROI Dice against the baseline and the fallbacks.
- `bootstrap` measures the noise floor of each method. It fits the tensor once and
  makes replicates by a wild bootstrap: each residual of the fit keeps its size and gets
  a random sign. For each replicate it refits the tensor in a central work region and
  reruns the ROI step; the registration stays fixed. It reports the SD and CV of ALPS,
  the L/R difference and the ROI Dice against the original fit.
- `summary` writes `summary_alps.csv`, `summary_regdiag.csv`, `summary_sensitivity.csv`,
  `summary_bootstrap.csv` and `tables.md`.

`alps/RESULTS.md` holds the current findings.

## Frames

- **b-vector frame.** DIPY fits the tensor in the frame of the b-vectors. The script does
  not reorient b-vectors.
- **World frame.** `M` maps the b-vector frame to world (RAS+) axes.
- **Template frame.** `R` is the rotation part (polar decomposition) of the affine that maps
  template world coordinates to native world coordinates. `Q = R^T M`.
- **Tilt.** `alps.json` reports the angle of `Q` after axis flips are removed. Axis flips do
  not change the diagonal elements.

## Data-driven ROI search

Both data-driven methods use the same search box in each hemisphere:

1. Find CSF-like voxels (MD > 2.0e-3 mm²/s, FA < 0.2) near the midline. Keep connected
   components larger than 500 mm³ as the lateral ventricles.
2. Set a prior box in template coordinates: y from -24 to 0 mm, z from 16 to 32 mm. Start x
   at the lateral wall of the ventricle body and extend it 30 mm. Limit z to the levels
   where the ventricle body exists.

### `data`

3. Give each voxel a soft weight per fibre group. The weight is the product of:
   - a falling sigmoid of the angle between the principal eigenvector and z (projection)
     or y (association), centred at 25°;
   - rising sigmoids of FA (centre 0.3) and linearity (centre 0.4);
   - falling sigmoids of planarity (centre 0.3) and MD (centre 1.1e-3 mm²/s, partial
     volume with CSF);
   - a weak Gaussian prior (σ = 6 mm) around the published MNI centre, for tie-breaks.
4. Choose the (y, z) centre whose 2.5 mm disc has the largest weight sum of the smaller
   group. Projection fibres must be medial to association fibres.
5. Grow each ROI from the highest-weight voxel of its group in that disc. At each step,
   add the highest-weight 26-connected neighbour. A new voxel needs a weight of at least
   0.1 and a principal eigenvector within 20° of the ROI mean. Growth stays within 5 mm of
   the centre in (y, z).
6. Stop at the voxel count of a 2.5 mm sphere on the native grid (8 voxels at 2 mm).
7. If the weight sum is below 1 in either group, use the `sphere` ROIs for that
   hemisphere and set `fallback` in `alps.json`. The index is never NaN.

Distances and the voxel count are in native mm, on template-aligned axes. The soft-score
centres and widths are in `SOFT_SCORES`. The other settings are keyword arguments of
`find_alps_rois_v2`. `alps.json` reports each ROI's weight sum, effective voxel count,
centroid, distance to the sphere centre and overlap with the sphere ROI.

### `data-legacy`

3. Select candidate voxels: FA ≥ 0.3, linearity ≥ 0.4, planarity ≤ 0.3. The principal
   eigenvector must be within 25° of z (projection) or y (association).
4. Choose the (y, z) centre whose 2.5 mm disc holds the most voxels of the smaller group.
   Projection fibres must be medial to association fibres.
5. In that disc, keep the candidates within 2.5 mm (along x) of each group's median x.

All thresholds are keyword arguments of `find_alps_rois`.

## Limitations

The script corrects head orientation and filters crossing fibres. It cannot correct these
limitations of the method:

- **Scale.** At b ≈ 1000 s/mm² the signal reflects water displacement of a few µm over tens
  of ms. Glymphatic transport is slow bulk flow. Perivascular spaces are a small fraction of
  each voxel.
- **Anatomical assumption.** The index assumes medullary veins run left-right and fibres
  run along y and z at the ventricle body. Anatomy varies between subjects and hemispheres.
- **Radial diffusivity.** `Dxx` in both ROIs is diffusivity perpendicular to the fibres.
  White-matter damage, fibre dispersion and crossing fibres change it.
- **Specificity.** The index decreases in many disorders and with age and vascular risk.
- **Acquisition.** The value depends on b-value, echo time, voxel size, number of
  directions and distortion correction. Compare values across sites only after
  harmonisation.
- **Validation.** Validation studies are few and small, and compare with other indirect
  measures. The output is named "ALPS index", not "glymphatic function".

## Tests

```bash
cd alps && python -m pytest test_alps.py
```

The tests use a synthetic phantom with known ALPS geometry. They check the rotation,
the FSL convention, the ROIs of each method, and the rejection of crossing fibres. For
`data` they also check the ROI size, the fallback on an isotropic phantom, and that the
weights change smoothly under a small rotation.

## References

- Taoka T, et al. Jpn J Radiol. 2017;35:172-178.
- Taoka T, et al. Magn Reson Med Sci. 2024;23:268-290.
- Park CJ, et al. npj Parkinsons Dis. 2026. doi:10.1038/s41531-026-01620-4
- Zhang W, et al. NeuroImage. 2021;238:118257.
- Schilling KG, et al. Hum Brain Mapp. 2025;46:e70282. <!-- codespell:ignore -->
- Ringstad G. Neuroradiology. 2024;66:157-160.
