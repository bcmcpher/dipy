# ALPS script: current findings

Status on 2026-11-12, branch `plan/alps`. All runs use DTI on b ≤ 1100 (Stanford: b = 2000
only). The tables after the findings come from `python alps/experiments.py summary`. Do not
edit them by hand; regenerate them and paste the new `alps_results/tables.md`.

## Datasets

| Name | Voxel size | Shells (b > 0) | Brain coverage | Note |
|---|---|---|---|---|
| `sherbrooke` | 2 mm | 1000, 2000, 3000 | z FOV 120 mm clips the brain | native tilt about 8° |
| `hbn` | 1.8 mm | 1000, 2000 (64 dirs each) | full | qsiprep, AC-PC aligned; paediatric |
| `stanford` | 2 mm | 2000 (150 dirs) | full | single high-b shell |

## Findings

1. **Registration.** The skull-stripped affine registration recovers an emulated tilt of
   ±10° with a rotation error ≤ 0.3° and an ROI-centre error ≤ 0.14 mm on HBN. On Sherbrooke
   the errors are larger (≤ 2.0°, ≤ 1.7 mm). Its clipped z FOV makes the affine compress the
   template along z (smallest singular value 0.73). On HBN the scale is near-isotropic
   (0.84–0.91), which matches a child's brain against the adult template.
2. **Rigid compared with affine.** A rigid-only registration gives larger rotation errors
   (up to 3.1° on Sherbrooke). The rotation therefore comes from the affine.
3. **Rotation correction.** On HBN, the sphere-ROI ALPS in the template frame varies by 1.7%
   (CV) across tilts from −10° to +10°. In the b-vector frame the CV is 3.1%. On Sherbrooke
   both values are larger (4.5% and 6.8%).
4. **Noise floor.** On Sherbrooke a ±0.5° tilt changes the sphere ALPS by up to 15%. A 5 mm
   sphere holds about 7 voxels at 2 mm, and the registration estimate of the tilt moves by
   about 1°. On HBN the same test changes ALPS by 1.5%.
5. **Data-driven ROIs.** The finder uses hard thresholds. It returns no ROI (NaN) in 2 of
   the 12 HBN tilt runs, and its L/R difference reaches 22%. This is the next work item.
6. **b-value.** Stanford (b = 2000) gives lower ALPS than HBN (b = 1000): 1.51 against 1.71
   for spheres. The datasets differ in subject and scanner too, so this is not a clean
   b-value effect.

## Open questions

- Data-driven ROI rework: soft-weighted ROIs, fixed-volume region growing, and RecoBundles
  CST/SLF bundles as an anatomical reference.
- Group comparisons are deferred until the feature set is complete and a comparison plan
  exists.


## ALPS index (direct runs, mean of L and R; L/R in brackets)

| dataset | measured tilt (deg) | sphere bvec | sphere template | data bvec | data template |
|---|---|---|---|---|---|
| sherbrooke | 8.33 | 1.838 (1.604/2.073) | 1.814 (1.653/1.975) | 1.371 (1.311/1.430) | 1.392 (1.102/1.681) |
| hbn | 3.95 | 1.738 (1.728/1.747) | 1.712 (1.692/1.731) | 1.561 (1.597/1.526) | 1.531 (1.556/1.507) |
| stanford | 2.42 | 1.498 (1.525/1.472) | 1.506 (1.539/1.473) | 1.358 (1.313/1.403) | 1.331 (1.298/1.364) |

## Tilt series: sherbrooke (about x; ALPS mean of L and R)

| tilt (deg) | measured tilt (deg) | sphere bvec | sphere template | data bvec | data template |
|---|---|---|---|---|---|
| +0.0 | 8.33 | 1.838 | 1.814 | 1.371 | 1.392 |
| +0.5 | 9.34 | 1.538 | 1.547 | 1.368 | 1.491 |
| -0.5 | 9.21 | 1.712 | 1.657 | 1.473 | 1.488 |
| +5.0 | 14.31 | 1.716 | 1.746 | 1.607 | 1.563 |
| -5.0 | 5.66 | 1.750 | 1.713 | 1.581 | 1.608 |
| +10.0 | 17.04 | 1.484 | 1.672 | 1.317 | 1.501 |
| -10.0 | 4.68 | 1.718 | 1.709 | 1.497 | 1.490 |
| CV across angles | | 6.8% | 4.5% | 7.1% | 4.2% |

## Tilt series: hbn (about x; ALPS mean of L and R)

| tilt (deg) | measured tilt (deg) | sphere bvec | sphere template | data bvec | data template |
|---|---|---|---|---|---|
| +0.0 | 3.96 | 1.738 | 1.712 | 1.561 | 1.531 |
| +0.5 | 4.19 | 1.765 | 1.737 | NaN | 1.495 |
| +5.0 | 8.33 | 1.700 | 1.681 | 1.635 | 1.613 |
| -5.0 | 3.16 | 1.655 | 1.668 | 1.462 | 1.614 |
| +10.0 | 13.26 | 1.634 | 1.657 | 1.503 | NaN |
| -10.0 | 7.31 | 1.629 | 1.710 | 1.526 | 1.602 |
| CV across angles | | 3.1% | 1.7% | 3.8% (1 NaN) | 3.1% (1 NaN) |

## Template registration under tilt (skull-stripped b0 and template)

Rotation error: angle of `R^T Rt R0`; ROI error: largest distance between
the mapped MNI ROI centres and the 0-deg centres rotated by the tilt.

| dataset | variant | tilt (deg) | rotation error (deg) | max ROI error (mm) | singular values | time (s) |
|---|---|---|---|---|---|---|
| sherbrooke | rigid | +0.0 | 0.0 | 0.0 | 1.0–1.0 | 22.7 |
| sherbrooke | rigid | +5.0 | 2.19 | 1.94 | 1.0–1.0 | 18.0 |
| sherbrooke | rigid | +10.0 | 0.69 | 1.85 | 1.0–1.0 | 34.4 |
| sherbrooke | rigid | -10.0 | 3.08 | 2.34 | 1.0–1.0 | 22.3 |
| sherbrooke | affine | +0.0 | 0.0 | 0.0 | 0.726–0.984 | 38.6 |
| sherbrooke | affine | +5.0 | 0.72 | 0.96 | 0.73–0.987 | 33.9 |
| sherbrooke | affine | +10.0 | 0.66 | 0.77 | 0.731–0.982 | 74.1 |
| sherbrooke | affine | -10.0 | 1.98 | 1.69 | 0.729–0.99 | 38.3 |
| hbn | affine | +0.0 | 0.0 | 0.0 | 0.844–0.907 | 54.6 |
| hbn | affine | +5.0 | 0.21 | 0.14 | 0.845–0.906 | 66.1 |
| hbn | affine | +10.0 | 0.18 | 0.12 | 0.845–0.907 | 70.5 |
| hbn | affine | -10.0 | 0.3 | 0.1 | 0.846–0.907 | 47.5 |
