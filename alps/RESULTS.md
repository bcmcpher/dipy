# ALPS script: current findings

Status on 2026-11-13, branch `plan/alps`. All runs use DTI on b ≤ 1100 (Stanford: b = 2000
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
5. **Data-driven ROIs, v1 (`data-legacy`).** The finder uses hard thresholds. It returns
   no ROI (NaN) in 1 of the 7 HBN tilt runs per frame, and its L/R difference ranges from
   −20% to +3% across tilts (template frame).
6. **Data-driven ROIs, v2 (`data`).** Soft weights, a shared centre and region growing.
   Growth stays within 5 mm of the seed in 3D (2 × `radius`).
   - No NaN and no fallback in any run: 3 direct runs, 7 HBN tilts, 41 sensitivity runs
     per dataset.
   - ROI size: 11 voxels per ROI on HBN (1.8 mm), 8 on Sherbrooke and Stanford (2 mm). One
     Stanford association ROI stops at 7 voxels: no more neighbours pass the angle limit.
   - HBN tilt CV (template frame) is 0.6%, against 1.6% for spheres and 3.5% for v1.
   - The L/R difference is stable across tilts (SD 2.3 points; spheres 3.6) but large:
     L is 18% lower than R on HBN. The left projection ROI has a higher Dyy (0.44e-3
     against 0.34e-3 mm²/s) and sits lower in z. Sherbrooke shows the opposite sign
     (L 1.73, R 1.49) and Stanford is symmetric (1.33 / 1.33), so this is subject anatomy
     or ROI placement, not a sign error. It needs a visual check.
   - Weighted and unweighted ALPS differ by ≤ 1.5%. Most of the gain comes from the
     centre choice and the region growing, not from the weights.
7. **Soft-score sensitivity (v2).** Each soft-score centre and width moved by ±10% and
   ±20%; fit and registration fixed. No perturbation causes a fallback.
   - Largest ALPS change at ±10%: HBN 3.3% (planarity centre ×1.1), Sherbrooke 3.5%
     (planarity width ×1.1), Stanford 3.0% (angle centre ×0.9). All three are at or just
     above the ≤ 3% criterion; HBN, the tuning set, misses it by 0.3 points.
   - At ±20% the largest change is 6.9% (HBN planarity centre). Planarity is the most
     sensitive score on every dataset.
   - With 8–11 voxel ROIs, one voxel swap changes ALPS by several percent.
   - MD has a small effect: ≤ 2.5% at ±10% on every dataset (HBN 0.8%). The MD score is
     not what places the ROI next to the ventricle; the box and the planarity and
     orientation scores do that.
8. **Noise floor (wild bootstrap, 50 replicates, registration fixed).** CV of the mean
   ALPS (template frame):

   | dataset | sphere | data-legacy | data |
   |---|---|---|---|
   | hbn | 1.1% | 3.2% (1 NaN) | 2.0% |
   | sherbrooke | 6.9% | 7.6% (1 NaN) | 7.5% |
   | stanford | 1.0% | 3.2% (10 NaN) | 3.6% (10 fallbacks) |

   - `data` is noisier than `sphere` under measurement noise: its ROI moves between
     replicates (median smallest Dice 0.45 on HBN, ~0 on Sherbrooke). The sphere ROI is
     fixed, so only the tensor noise reaches it.
   - Against this floor, the v2 tilt CV (HBN 0.6%, Sherbrooke 3.3%) is below the noise,
     which meets the "tilt CV ≤ bootstrap CV" criterion. The 3.3% HBN sensitivity at ±10%
     is about 1.6 bootstrap SDs, so the 0.3-point miss of the 3% criterion is within noise.
   - The HBN L/R difference of `data` stays at −15.6% (SD 4.1) across replicates. It is
     systematic (anatomy or placement), not noise.
   - Stanford: the left hemisphere falls back to the sphere ROIs in 10 of 50 replicates.
     Its association group is weak (7-voxel ROI at baseline). v1 returns NaN in the same
     number of replicates.
   - Sherbrooke is noisy for every method (~7%). Its 2 mm b = 1000 shell gives low SNR in
     8-voxel ROIs; single-subject values there are not reliable.
9. **Sherbrooke tilt series (v2).** Template-frame CV 3.3%, against 4.5% for spheres and
   4.2% for v1; all below its 7.5% noise floor.
10. **b-value.** Stanford (b = 2000) gives lower ALPS than HBN (b = 1000): 1.51 against 1.71
   for spheres. The datasets differ in subject and scanner too, so this is not a clean
   b-value effect.

## Open questions

- Visual audit of the v2 ROIs on HBN (left projection ROI low in z).
- `data` trades fixed-ROI noise robustness for placement: its bootstrap CV is about
  twice the sphere's on HBN and Stanford. Larger ROIs (`radius`), or averaging the
  weights over replicates, may reduce it.
- Stanford left-hemisphere fallbacks under noise (20% of replicates).
- Periventricular white-matter hyperintensities raise MD and may lower the MD weight in
  damaged tissue, which would bias ALPS upwards in patients. Free-water fraction
  (Phase 2) is the planned alternative to the MD score.
- Group comparisons are deferred until the feature set is complete and a comparison plan
  exists.

## ALPS index (direct runs, mean of L and R; L/R in brackets)

| dataset | measured tilt (deg) | sphere bvec | sphere template | data-legacy bvec | data-legacy template | data bvec | data template |
|---|---|---|---|---|---|---|---|
| sherbrooke | 8.33 | 1.838 (1.604/2.073) | 1.814 (1.653/1.975) | 1.371 (1.311/1.430) | 1.392 (1.102/1.681) | 1.516 (1.422/1.610) | 1.609 (1.727/1.491) |
| hbn | 3.95 | 1.738 (1.728/1.747) | 1.712 (1.692/1.731) | 1.561 (1.597/1.526) | 1.531 (1.556/1.507) | 1.611 (1.428/1.793) | 1.549 (1.393/1.704) |
| stanford | 2.42 | 1.498 (1.525/1.472) | 1.506 (1.539/1.473) | 1.358 (1.313/1.403) | 1.331 (1.298/1.364) | 1.347 (1.347/1.347) | 1.328 (1.329/1.327) |

## Tilt series: sherbrooke (about x; ALPS mean of L and R)

| tilt (deg) | measured tilt (deg) | sphere bvec | sphere template | data-legacy bvec | data-legacy template | data bvec | data template |
|---|---|---|---|---|---|---|---|
| +0.0 | 8.33 | 1.838 | 1.814 | 1.371 | 1.392 | 1.516 | 1.609 |
| +0.5 | 9.34 | 1.538 | 1.547 | 1.368 | 1.491 | 1.358 | 1.521 |
| -0.5 | 9.21 | 1.712 | 1.657 | 1.473 | 1.488 | 1.465 | 1.603 |
| +5.0 | 14.31 | 1.716 | 1.746 | 1.607 | 1.563 | 1.641 | 1.601 |
| -5.0 | 5.66 | 1.750 | 1.713 | 1.581 | 1.608 | 1.606 | 1.493 |
| +10.0 | 17.04 | 1.484 | 1.672 | 1.317 | 1.501 | 1.452 | 1.562 |
| -10.0 | 4.68 | 1.718 | 1.709 | 1.497 | 1.490 | 1.516 | 1.476 |
| CV across angles | | 6.8% | 4.5% | 7.1% | 4.2% | 5.9% | 3.3% |

## Tilt series: hbn (about x; ALPS mean of L and R)

| tilt (deg) | measured tilt (deg) | sphere bvec | sphere template | data-legacy bvec | data-legacy template | data bvec | data template |
|---|---|---|---|---|---|---|---|
| +0.0 | 3.96 | 1.738 | 1.712 | 1.561 | 1.531 | 1.611 | 1.548 |
| +0.5 | 4.19 | 1.765 | 1.737 | NaN | 1.495 | 1.650 | 1.559 |
| -0.5 | 3.42 | 1.743 | 1.720 | 1.698 | 1.659 | 1.618 | 1.573 |
| +5.0 | 8.33 | 1.700 | 1.681 | 1.635 | 1.613 | 1.660 | 1.551 |
| -5.0 | 3.16 | 1.655 | 1.668 | 1.462 | 1.614 | 1.573 | 1.560 |
| +10.0 | 13.26 | 1.634 | 1.657 | 1.503 | NaN | 1.631 | 1.547 |
| -10.0 | 7.31 | 1.629 | 1.710 | 1.526 | 1.602 | 1.466 | 1.548 |
| CV across angles | | 3.1% | 1.6% | 5.1% (1 NaN) | 3.5% (1 NaN) | 3.8% | 0.6% |

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

## Soft-score sensitivity: sherbrooke (`data`, template frame)

Baseline ALPS 1.609 (L 1.727, R 1.491). Cells: change of the mean ALPS in % (smallest ROI Dice against the baseline; F = number of hemispheres that fell back).

| score | part | x0.8 | x0.9 | x1.1 | x1.2 |
|---|---|---|---|---|---|
| angle | centre | +2.5 (0.50) | +1.0 (0.62) | -2.4 (0.88) | -2.9 (0.75) |
| angle | width | -2.3 (0.88) | -2.3 (0.88) | +0.0 (1.00) | +0.0 (1.00) |
| fa | centre | -0.0 (1.00) | -0.0 (1.00) | +0.0 (1.00) | -2.1 (0.88) |
| fa | width | -0.0 (1.00) | -0.0 (1.00) | +0.0 (1.00) | +0.0 (1.00) |
| cl | centre | +0.3 (0.88) | +0.7 (0.88) | -1.7 (0.88) | -1.9 (0.62) |
| cl | width | +0.7 (0.88) | -0.1 (0.88) | -2.1 (0.88) | -2.1 (0.88) |
| cp | centre | -2.2 (0.62) | -1.3 (0.62) | +0.2 (0.88) | -0.5 (0.75) |
| cp | width | -0.2 (0.88) | +0.0 (0.88) | +3.5 (0.75) | +3.4 (0.75) |
| md | centre | -0.7 (0.50) | -2.1 (0.88) | +0.1 (0.62) | +0.1 (0.62) |
| md | width | +0.1 (0.62) | -0.0 (1.00) | +0.0 (1.00) | -2.1 (0.88) |

## Soft-score sensitivity: hbn (`data`, template frame)

Baseline ALPS 1.549 (L 1.393, R 1.704). Cells: change of the mean ALPS in % (smallest ROI Dice against the baseline; F = number of hemispheres that fell back).

| score | part | x0.8 | x0.9 | x1.1 | x1.2 |
|---|---|---|---|---|---|
| angle | centre | +1.1 (0.82) | -0.3 (0.91) | -0.9 (0.45) | -1.3 (0.36) |
| angle | width | -0.0 (0.91) | -0.0 (0.91) | -0.0 (1.00) | -0.0 (1.00) |
| fa | centre | -0.1 (1.00) | -0.0 (1.00) | +0.1 (0.91) | +1.7 (0.91) |
| fa | width | -0.1 (1.00) | -0.0 (1.00) | +0.0 (0.91) | +0.1 (0.91) |
| cl | centre | -1.8 (0.73) | -0.8 (0.73) | +1.1 (0.91) | +2.4 (0.45) |
| cl | width | -0.2 (0.91) | -0.1 (0.91) | +0.1 (0.91) | +0.1 (0.91) |
| cp | centre | -4.5 (0.36) | -1.8 (0.45) | +3.3 (0.91) | +6.9 (0.82) |
| cp | width | +1.4 (0.91) | -0.1 (0.91) | +0.1 (0.91) | +0.1 (0.91) |
| md | centre | +1.4 (0.64) | -0.5 (0.91) | -0.8 (0.91) | -0.7 (0.82) |
| md | width | +0.0 (0.91) | +0.1 (0.91) | -0.1 (0.91) | -0.1 (0.91) |

## Soft-score sensitivity: stanford (`data`, template frame)

Baseline ALPS 1.328 (L 1.329, R 1.327). Cells: change of the mean ALPS in % (smallest ROI Dice against the baseline; F = number of hemispheres that fell back).

| score | part | x0.8 | x0.9 | x1.1 | x1.2 |
|---|---|---|---|---|---|
| angle | centre | +4.5 (0.12) | +3.0 (0.12) | -0.2 (1.00) | -0.3 (1.00) |
| angle | width | +2.5 (0.12) | +2.5 (0.12) | +0.0 (1.00) | +0.0 (1.00) |
| fa | centre | -0.0 (1.00) | -0.0 (1.00) | +2.5 (0.12) | +2.6 (0.12) |
| fa | width | -0.0 (1.00) | -0.0 (1.00) | +2.5 (0.12) | +2.5 (0.12) |
| cl | centre | +2.1 (0.75) | +0.3 (1.00) | +1.9 (0.12) | +2.3 (0.12) |
| cl | width | +0.0 (1.00) | -0.0 (1.00) | +0.0 (1.00) | +0.0 (1.00) |
| cp | centre | +0.4 (0.12) | +1.3 (0.12) | +0.5 (1.00) | +2.2 (0.88) |
| cp | width | +2.6 (0.12) | -0.0 (1.00) | +0.0 (1.00) | +0.0 (1.00) |
| md | centre | +2.5 (0.12) | +2.5 (0.12) | +0.0 (1.00) | +0.0 (1.00) |
| md | width | +0.0 (1.00) | +0.0 (1.00) | -0.0 (1.00) | -0.0 (1.00) |

## Wild bootstrap: sherbrooke (50 replicates; registration fixed)

Replicate 0 is the fit to the data. SD and CV over the replicates; L-R: L/R difference in % (mean and SD); Dice: median of the smallest ROI Dice against replicate 0.

| method | frame | ALPS (rep 0) | SD | CV | CV L | CV R | L-R mean (SD) | Dice | fallbacks |
|---|---|---|---|---|---|---|---|---|---|
| sphere | bvec | 1.838 | 0.129 | 7.1% | 8.1% | 9.9% | -27.2 (11.2) | 1.00 | 0 |
| sphere | template | 1.814 | 0.122 | 6.9% | 8.2% | 9.4% | -19.8 (11.0) | 1.00 | 0 |
| data-legacy | bvec | 1.371 | 0.144 | 10.2% | 14.9% | 13.2% | -24.2 (18.2) | 0.00 | 0 |
| data-legacy | template | 1.392 | 0.104 | 7.6% (1 NaN) | 12.2% | 10.9% | -15.4 (17.0) | 0.00 | 0 |
| data | bvec | 1.516 | 0.095 | 6.7% | 11.0% | 7.8% | -12.8 (13.0) | 0.12 | 0 |
| data | template | 1.609 | 0.102 | 7.5% | 11.2% | 8.5% | -8.1 (13.1) | 0.00 | 0 |

## Wild bootstrap: hbn (50 replicates; registration fixed)

Replicate 0 is the fit to the data. SD and CV over the replicates; L-R: L/R difference in % (mean and SD); Dice: median of the smallest ROI Dice against replicate 0.

| method | frame | ALPS (rep 0) | SD | CV | CV L | CV R | L-R mean (SD) | Dice | fallbacks |
|---|---|---|---|---|---|---|---|---|---|
| sphere | bvec | 1.738 | 0.018 | 1.1% | 1.2% | 1.8% | -1.5 (2.2) | 1.00 | 0 |
| sphere | template | 1.712 | 0.018 | 1.1% | 1.2% | 1.8% | -2.6 (2.3) | 1.00 | 0 |
| data-legacy | bvec | 1.561 | 0.052 | 3.2% | 4.1% | 4.9% | -10.2 (6.5) | 0.00 | 0 |
| data-legacy | template | 1.531 | 0.050 | 3.2% (1 NaN) | 4.1% | 4.6% | -11.4 (6.2) | 0.00 | 0 |
| data | bvec | 1.611 | 0.036 | 2.2% | 2.9% | 3.4% | -14.2 (4.4) | 0.45 | 0 |
| data | template | 1.549 | 0.030 | 2.0% | 2.6% | 3.1% | -15.6 (4.1) | 0.45 | 0 |

## Wild bootstrap: stanford (50 replicates; registration fixed)

Replicate 0 is the fit to the data. SD and CV over the replicates; L-R: L/R difference in % (mean and SD); Dice: median of the smallest ROI Dice against replicate 0.

| method | frame | ALPS (rep 0) | SD | CV | CV L | CV R | L-R mean (SD) | Dice | fallbacks |
|---|---|---|---|---|---|---|---|---|---|
| sphere | bvec | 1.498 | 0.015 | 1.0% | 1.5% | 1.7% | +3.5 (2.6) | 1.00 | 0 |
| sphere | template | 1.506 | 0.015 | 1.0% | 1.5% | 1.7% | +4.3 (2.6) | 1.00 | 0 |
| data-legacy | bvec | 1.358 | 0.035 | 2.5% (10 NaN) | 3.8% | 3.4% | -0.3 (5.3) | 0.40 | 0 |
| data-legacy | template | 1.331 | 0.043 | 3.2% (10 NaN) | 5.5% | 2.9% | -1.4 (6.1) | 0.20 | 0 |
| data | bvec | 1.347 | 0.045 | 3.2% | 5.9% | 2.0% | +1.8 (5.8) | 0.62 | 10 |
| data | template | 1.328 | 0.049 | 3.6% | 6.8% | 1.8% | +3.0 (6.8) | 0.69 | 10 |
