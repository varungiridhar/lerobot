#!/bin/bash
# Download MG source HDF5s for square + threading + coffee.
# Login node was hitting 404 on drive.usercontent.google.com; compute node
# may take a different egress path.
set -euxo pipefail

TARGET="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/source/raw"
mkdir -p "${TARGET}"
cd "${TARGET}"

PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mg-datagen/bin/python"

"${PYBIN}" -u -m mimicgen.scripts.download_datasets \
    --tasks square threading coffee \
    --dataset_type source \
    --download_dir "${TARGET}"

echo "---"
ls -la "${TARGET}"
find "${TARGET}" -name '*.hdf5' -exec ls -la {} \;
