"""Local test datasets for the ALPS experiments.

Each entry names a DWI (``.nii.gz``, ``.bval``, ``.bvec`` with a common stem)
already in the DIPY data cache, plus the b-value cut used for DTI.
"""

from pathlib import Path

DIPY_HOME = Path.home() / ".dipy"

DATASETS = {
    # 2 mm, b = 1000/2000/3000; z FOV 120 mm clips the brain.
    "sherbrooke": {
        "stem": DIPY_HOME / "sherbrooke_3shell" / "HARDI193",
        "max_bval": 1100.0,
    },
    # qsiprep output (T1w/AC-PC space), 1.8 mm, b = 1000/2000, 64 dirs each.
    "hbn": {
        "stem": DIPY_HOME
        / "HBN/derivatives/qsiprep/sub-NDARAA948VFH/ses-HBNsiteRU/dwi"
        / "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi",
        "max_bval": 1100.0,
    },
    # 2 mm, single shell b = 2000, 150 dirs.
    "stanford": {
        "stem": DIPY_HOME / "stanford_hardi" / "HARDI150",
        "max_bval": None,
    },
}


def dataset_paths(name):
    """Return ``(dwi, bval, bvec, max_bval)`` for a named dataset."""
    entry = DATASETS[name]
    stem = str(entry["stem"])
    paths = (f"{stem}.nii.gz", f"{stem}.bval", f"{stem}.bvec")
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Dataset {name!r} is incomplete: {missing}")
    return (*paths, entry["max_bval"])
