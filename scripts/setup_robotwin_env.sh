#!/bin/bash
# =============================================================================
# RoboTwin 2.0 environment setup for LeRobot.
#
# Strategy: CLONE the existing working `lerobot` conda env (LeRobot 0.4.4 +
# torch 2.7.1 / gymnasium 1.x, already fully resolved) and layer the RoboTwin
# SAPIEN simulator dependencies on top. A from-scratch `pip install -e .` of
# LeRobot hits an unresolvable pip backtrack (imageio extra); cloning the
# known-good env sidesteps that entirely. The clone is a separate env, so the
# original `lerobot` env is left untouched (the requested isolation).
#
# RoboTwin's requirements.txt pins torch==2.4.1 / gymnasium==0.29.1; those pins
# are intentionally dropped — its sim libs are torch-version agnostic, matching
# how the upstream FastWAM repo (torch==2.7.1) runs RoboTwin.
#
# Submit as a SLURM job (GPU node needed for the Vulkan render test):
#     sbatch scripts/setup_robotwin_env.sh
# Heavy CUDA-compiled deps (curobo, pytorch3d) are gated; enable with:
#     WITH_PLANNERS=1 sbatch scripts/setup_robotwin_env.sh
# =============================================================================
#SBATCH --job-name=robotwin-setup
#SBATCH --account=gts-agarg35
#SBATCH -N1
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=48G
#SBATCH -q embers
#SBATCH -t 4:00:00
#SBATCH --output=slurm_out/Report-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vgiridhar6@gatech.edu

set -uxo pipefail   # NOTE: no -e — the diagnostic block must run to completion.

# ---- Paths (override via environment variables) -----------------------------
WORKTREE="${WORKTREE:-/storage/home/hcoda1/6/vgiridhar6/forks/lerobot-robotwin}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/storage/home/hcoda1/6/vgiridhar6/r-agarg35-0/robotwin/RoboTwin}"
SRC_ENV="${SRC_ENV:-/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot}"
CONDA_ENV="${CONDA_ENV:-/storage/project/r-agarg35-0/vgiridhar6/.conda/envs/lerobot-robotwin}"
SCRATCH="${SCRATCH:-/storage/project/r-agarg35-0/vgiridhar6/robotwin/tmp}"
WITH_PLANNERS="${WITH_PLANNERS:-0}"   # 1 → also build curobo + pytorch3d (slow)

mkdir -p "$SCRATCH"
export TMPDIR="$SCRATCH"
export PIP_CACHE_DIR="$SCRATCH/pip-cache"
export PYTHONUNBUFFERED=1

module load anaconda3/2022.05.0.1
module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- 1. Clone the working lerobot env ---------------------------------------
if python -c "import sys; sys.path.insert(0,'$CONDA_ENV/lib/python3.10/site-packages'); import lerobot" 2>/dev/null \
   && [ -d "$CONDA_ENV" ]; then
    echo "### Conda env $CONDA_ENV already has lerobot — reusing"
else
    echo "### Removing any partial env and cloning $SRC_ENV -> $CONDA_ENV"
    conda env remove -p "$CONDA_ENV" -y 2>/dev/null || true
    rm -rf "$CONDA_ENV"
    conda create --clone "$SRC_ENV" -p "$CONDA_ENV" -y
fi
conda activate "$CONDA_ENV"
python --version
which python

# ---- 2. Repoint the editable lerobot install at this worktree ---------------
# --no-deps: all deps are already present in the cloned env; this only rebinds
# the `lerobot` editable package to the robotwin worktree's source tree.
echo "### Repointing editable lerobot at $WORKTREE"
cd "$WORKTREE"
python -m pip install -e . --no-deps
python -m pip install "h5py>=3.10"   # used by dataset tooling; not in lerobot core

# ---- 3. RoboTwin SAPIEN sim stack -------------------------------------------
# Installed in its own resolve. numpy is NOT pinned here: we first try to keep
# the cloned env's numpy 2.x; the diagnostic block below reports if a sim lib
# forced a downgrade and whether the scientific stack survived it.
echo "### Installing RoboTwin SAPIEN sim stack"
python -m pip install \
    "sapien==3.0.0b1" \
    "mplib==0.2.1" \
    "toppra" \
    "transforms3d==0.4.2" \
    "trimesh==4.4.3" \
    "open3d" \
    "pyglet<2"

# ---- 4. CUDA-compiled motion-planning deps ----------------------------------
# curobo is REQUIRED: RoboTwin's envs/robot/robot.py hard-imports CuroboPlanner,
# which is undefined unless curobo imports cleanly — so without it no RoboTwin
# task can even be imported. pytorch3d is genuinely optional (point-cloud FPS in
# a try/except) and is skipped unless WITH_PYTORCH3D=1.
if [ "$WITH_PLANNERS" = "1" ]; then
    export FORCE_CUDA=1
    export MAX_JOBS=6
    CUROBO_DIR="$ROBOTWIN_ROOT/envs/curobo"
    if ! python -c "import curobo" 2>/dev/null; then
        echo "### Building curobo v0.7.8 from source (CUDA compile — slow)"
        [ -d "$CUROBO_DIR" ] || git clone --branch v0.7.8 --depth 1 \
            https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
        python -m pip install -e "$CUROBO_DIR" --no-build-isolation
    else
        echo "### curobo already importable — skipping"
    fi
    # curobo's setup.cfg pins only `warp-lang>=0.9.0`, so the build pulls the
    # latest warp — but warp >=~1.5 dropped the auto-imported `wp.torch`
    # attribute that curobo 0.7.8 relies on (AttributeError in world_mesh.py).
    # Pin warp-lang to a curobo-0.7.8-contemporaneous release.
    python -m pip install "warp-lang==1.3.0"
    if [ "${WITH_PYTORCH3D:-0}" = "1" ]; then
        python -c "import pytorch3d" 2>/dev/null || \
            python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
                --no-build-isolation
    fi
else
    echo "### Skipping curobo (WITH_PLANNERS=0) — RoboTwin task import will fail without it."
fi

# ---- 5. Patch sapien / mplib (per RoboTwin script/_install.sh) ---------------
SAPIEN_DIR="$(python -c 'import os,sapien; print(os.path.dirname(sapien.__file__))' 2>/dev/null || true)"
[ -n "$SAPIEN_DIR" ] && [ -f "$SAPIEN_DIR/wrapper/urdf_loader.py" ] && \
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SAPIEN_DIR/wrapper/urdf_loader.py" && \
    echo "### Patched sapien urdf_loader.py"
MPLIB_DIR="$(python -c 'import os,mplib; print(os.path.dirname(mplib.__file__))' 2>/dev/null || true)"
[ -n "$MPLIB_DIR" ] && [ -f "$MPLIB_DIR/planner.py" ] && \
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$MPLIB_DIR/planner.py" && \
    echo "### Patched mplib planner.py"

# ---- 6. Diagnostic import block (non-fatal) ---------------------------------
echo "### ===== DIAGNOSTIC ====="
python - <<'PY'
def chk(name, mod=None):
    mod = mod or name
    try:
        m = __import__(mod)
        print(f"[ OK ] {name:18s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"[FAIL] {name:18s} {type(e).__name__}: {e}")

import importlib.metadata as md
try:
    print("numpy meta version:", md.version("numpy"))
except Exception as e:
    print("numpy meta:", e)

for n in ["numpy", "scipy", "pandas", "torch", "torchvision",
          "gymnasium", "sapien", "mplib", "toppra", "transforms3d",
          "trimesh", "open3d", "transformers", "diffusers"]:
    chk(n)
chk("opencv (cv2)", "cv2")

try:
    import lerobot
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy  # noqa
    print("[ OK ] lerobot + fastwam import")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[FAIL] lerobot/fastwam: {e}")
PY

echo "### Importing a RoboTwin task class"
cd "$ROBOTWIN_ROOT"
python - <<'PY'
import sys, os, traceback
sys.path.insert(0, os.getcwd())
try:
    import envs.beat_block_hammer as m
    inst = getattr(m, "beat_block_hammer")()
    print("[ OK ] RoboTwin task import + instantiate:", type(inst).__name__)
except Exception as e:
    traceback.print_exc()
    print(f"[FAIL] RoboTwin task import: {e}")
PY

# ---- 7. SAPIEN headless render de-risk (TOP RISK) ---------------------------
echo "### SAPIEN render test (script/test_render.py)"
python script/test_render.py 2>&1 | tee "$SCRATCH/test_render.log" || true
if grep -q "Render Well" "$SCRATCH/test_render.log"; then
    echo "### SAPIEN RENDER: OK"
else
    echo "### SAPIEN RENDER: FAILED — inspect log (Vulkan/ICD issue likely)"
fi

echo "### setup_robotwin_env.sh complete"
