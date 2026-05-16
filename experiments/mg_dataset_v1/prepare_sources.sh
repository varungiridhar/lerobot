#!/bin/bash
# Add MimicGen datagen metadata to threading.hdf5 and coffee.hdf5.
# prepare_src_dataset.py modifies in-place, so we copy raw → prepared first.
#
# Submit through compute_rtx6000_mimicgen.sh:
#   sbatch scripts/compute_rtx6000_mimicgen.sh \
#       bash experiments/mg_dataset_v1/prepare_sources.sh
set -euxo pipefail

RAW_DIR="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/source/raw"
PREP_DIR="/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/source/prepared"
mkdir -p "${PREP_DIR}"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
PYBIN="/storage/home/hcoda1/6/vgiridhar6/.conda/envs/lerobot-mg-datagen/bin/python"

cd /storage/home/hcoda1/6/vgiridhar6/forks/lerobot

prep_one() {
    local task=$1
    local interface=$2
    local raw="${RAW_DIR}/${task}.hdf5"
    local out="${PREP_DIR}/${task}.hdf5"
    if [[ ! -f "${raw}" ]]; then echo "missing ${raw}"; exit 1; fi
    cp "${raw}" "${out}"
    "${PYBIN}" -u -m mimicgen.scripts.prepare_src_dataset \
        --dataset "${out}" \
        --env_interface "${interface}" \
        --env_interface_type robosuite
}

prep_one threading MG_Threading
prep_one coffee    MG_Coffee
