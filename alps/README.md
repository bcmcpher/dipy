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
   It rejects voxels with low linearity or high planarity, which removes crossing fibres.
   The published 5 mm spheres are kept as the reference method.
4. **Quality metrics.** Each ROI reports its voxel count, FA, MD, linearity, planarity
   and the mean angle between the principal eigenvector and the expected fibre axis.

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
- `--roi-method {sphere,data,both}`: ROI method. Default `both`.
- `--reg {affine,syn}`: registration used to map ROIs. The rotation always comes from the
  affine step.

Outputs in `--out-dir`:

| File | Content |
|---|---|
| `alps.csv` | One row per method, frame and hemisphere: ALPS, the diagonal diffusivities and the ROI metrics |
| `alps.json` | Rotations (`M`, `R`, `Q`), head tilt, registration affine, data-driven search details |
| `rois_<method>.nii.gz` | ROI labels: 1 L proj, 2 L assoc, 3 R proj, 4 R assoc |
| `qc_alps.png` | ROIs on colour FA (template frame), ventricle outline, registration check |
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
python alps/experiments.py summary
```

- `direct` runs `alps.py` on the dataset.
- `tilt` rotates the anatomy and the b-vectors about the image centre, which emulates a
  tilted head, and reruns the full pipeline at each angle. It deletes the tilted DWIs
  unless you pass `--keep-dwi`.
- `regdiag` registers a tilted mean b0 to the template. It reports the rotation error and
  the ROI placement error against the known tilt.
- `summary` writes `summary_alps.csv`, `summary_regdiag.csv` and `tables.md`.

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

Each hemisphere uses this procedure:

1. Find CSF-like voxels (MD > 2.0e-3 mm²/s, FA < 0.2) near the midline. Keep connected
   components larger than 500 mm³ as the lateral ventricles.
2. Set a prior box in template coordinates: y from -24 to 0 mm, z from 16 to 32 mm. Start x
   at the lateral wall of the ventricle body and extend it 30 mm. Limit z to the levels
   where the ventricle body exists.
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
the FSL convention, the sphere and data-driven ROIs, and the rejection of crossing fibres.

## References

- Taoka T, et al. Jpn J Radiol. 2017;35:172-178.
- Taoka T, et al. Magn Reson Med Sci. 2024;23:268-290.
- Park CJ, et al. npj Parkinsons Dis. 2026. doi:10.1038/s41531-026-01620-4
- Zhang W, et al. NeuroImage. 2021;238:118257.
- Schilling KG, et al. Hum Brain Mapp. 2025;46:e70282. <!-- codespell:ignore -->
- Ringstad G. Neuroradiology. 2024;66:157-160.
