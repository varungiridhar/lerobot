"""Download LIBERO datasets to shared fast storage.

Usage:
    python scripts/download_libero_data.py

Submit via sbatch for large downloads (pure network I/O, no GPU needed).
"""

from huggingface_hub import snapshot_download

DEST_ROOT = "/storage/home/hcoda1/7/igeorgiev3/shared/lerobot-data-2"

DATASETS = [
    "HuggingFaceVLA/libero",
    "yilin-wu/libero-100",
]

for repo_id in DATASETS:
    local_dir = f"{DEST_ROOT}/{repo_id}"
    print(f"Downloading {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )
    print(f"Done: {repo_id}")
