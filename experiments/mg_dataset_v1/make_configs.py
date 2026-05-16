"""Build MimicGen configs for the v1 dataset (square / threading / coffee × q5 / q3_termjitter).

Output configs live in:
  /storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play/configs/
    square_q5.json
    square_q3_termjitter.json
    threading_q5.json
    threading_q3_termjitter.json
    coffee_q5.json
    coffee_q3_termjitter.json

Each config inherits from the per-task MG template under
  /storage/home/hcoda1/6/vgiridhar6/forks/mimicgen/mimicgen/exps/templates/robosuite/<task>.json
and overrides:
  * source dataset path  → prepared/<task>.hdf5
  * generation path      → v1 dataset / <task> / <bucket>
  * num_trials           → 200, keep_failed=True, guarantee=False
  * obs.camera_names     → ['agentview', 'robot0_eye_in_hand'] @ 84x84
  * task name            → <Task>_D0
  * per-bucket subtask overrides

q3_termjitter range is task-tuned: square's grasp default range is [10,20]
so we cut [-15,-5] before the signal; threading and coffee default to [5,10]
so we use the smaller [-10,-3] window.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path("/storage/home/hcoda1/6/vgiridhar6/forks/lerobot")
MG_TEMPLATES = Path(
    "/storage/home/hcoda1/6/vgiridhar6/forks/mimicgen/mimicgen/exps/templates/robosuite"
)
DATASET_ROOT = Path(
    "/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/lerobot_datasets/v1_q5_q3jitter_play"
)
SOURCE_DIR = Path("/storage/home/hcoda1/6/vgiridhar6/shared/mimicgen/source/prepared")
CONFIGS_DIR = DATASET_ROOT / "configs"

TASKS = {
    "square":    {"template": "square.json",    "task_name": "Square_D0",    "termjitter_range": [-15, -5]},
    "threading": {"template": "threading.json", "task_name": "Threading_D0", "termjitter_range": [-10, -3]},
    "coffee":    {"template": "coffee.json",    "task_name": "Coffee_D0",    "termjitter_range": [-10, -3]},
}

BUCKETS = ("q5", "q3_termjitter")
NUM_TRIALS = 200
SEED_BASE = 1000


def build_one(task: str, bucket: str, idx: int) -> dict:
    spec = TASKS[task]
    cfg = json.loads((MG_TEMPLATES / spec["template"]).read_text())

    cfg["experiment"]["name"] = bucket
    cfg["experiment"]["source"]["dataset_path"] = str(SOURCE_DIR / f"{task}.hdf5")
    cfg["experiment"]["generation"]["path"] = str(DATASET_ROOT / task / bucket)
    cfg["experiment"]["generation"]["num_trials"] = NUM_TRIALS
    cfg["experiment"]["generation"]["keep_failed"] = True
    cfg["experiment"]["generation"]["guarantee"] = False
    cfg["experiment"]["task"]["name"] = spec["task_name"]
    cfg["experiment"]["seed"] = SEED_BASE + idx
    cfg["experiment"]["num_demo_to_render"] = 10
    cfg["experiment"]["num_fail_demo_to_render"] = 5
    cfg["experiment"]["max_num_failures"] = NUM_TRIALS

    cfg["obs"]["collect_obs"] = True
    cfg["obs"]["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    cfg["obs"]["camera_height"] = 84
    cfg["obs"]["camera_width"] = 84

    if bucket == "q5":
        # Clean expert: zero action noise, default termination range.
        for s in ("subtask_1", "subtask_2"):
            cfg["task"]["task_spec"][s]["action_noise"] = 0.0
            cfg["task"]["task_spec"][s]["apply_noise_during_interpolation"] = False
    elif bucket == "q3_termjitter":
        # Premature grasp release: subtask_1 ends some steps BEFORE the grasp
        # signal would normally fire. Trajectory shape stays smooth, the timing
        # of release is wrong → mirrors a teleop slip.
        cfg["task"]["task_spec"]["subtask_1"]["action_noise"] = 0.0
        cfg["task"]["task_spec"]["subtask_1"]["subtask_term_offset_range"] = (
            spec["termjitter_range"]
        )
        cfg["task"]["task_spec"]["subtask_2"]["action_noise"] = 0.0
        cfg["task"]["task_spec"]["subtask_2"]["apply_noise_during_interpolation"] = False
    else:
        raise ValueError(bucket)

    return cfg


def main() -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    idx = 0
    for task in TASKS:
        for bucket in BUCKETS:
            cfg = build_one(task, bucket, idx)
            out_path = CONFIGS_DIR / f"{task}_{bucket}.json"
            out_path.write_text(json.dumps(cfg, indent=4))
            print(
                f"wrote {out_path.name}  seed={SEED_BASE + idx}  "
                f"src={cfg['experiment']['source']['dataset_path']}"
            )
            idx += 1


if __name__ == "__main__":
    main()
