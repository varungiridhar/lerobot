Scripts your collaborator runs

  Q training (one task, ~30 min on RTX 6000)

  The experiments/mg_dataset_v1/ launcher is not in the repo per your instructions. They'll need to either:
  - Get a copy of experiments/mg_dataset_v1/run_q_train_lerobot.sh from you (it orchestrates: convert HDF5 → LeRobotDataset → precache → train), OR
  - Invoke the steps directly. Minimal sequence (all from repo root):

  # 1. Convert HDF5 buckets → LeRobotDatasets (one-time per task)
  #    (script also lives in experiments/mg_dataset_v1/; not in repo)
  python experiments/mg_dataset_v1/convert_v1_to_lerobot.py \
      --src_root <SHARED>/v1_q5_q3jitter_play \
      --dst_root <SHARED>/v1_lerobot \
      --task threading --bucket q5  # repeat for q3_termjitter, play

  # 2. Pre-cache features for each bucket through the BC backbone
  python -m lerobot.policies.act_simple.precache_features \
      --policy_checkpoint outputs/train/mimicgen_threading_d0_act_simple/checkpoints/100000/pretrained_model \
      --dataset_root <SHARED>/v1_lerobot/mg_threading_q5 \
      --repo_id mg_threading_q5   # repeat for q3_termjitter, play

  # 3. Train Q
  python -m lerobot.scripts.lerobot_train \
      --dataset.repo_ids="[mg_threading_q5,mg_threading_q3_termjitter,mg_threading_play]" \
      --dataset.root=<SHARED>/v1_lerobot \
      --dataset.use_imagenet_stats=false \
      --policy.type=q_function \
      --policy.h=10 \
      --policy.vision_backbone=resnet18_cached \
      --policy.precache_root=outputs/train/mimicgen_threading_d0_act_simple/checkpoints/100000/pretrained_model/encoded_backbone \
      --policy.push_to_hub=false \
      --policy.device=cuda \
      --output_dir=outputs/train/mimicgen_threading_d0_q_function_h10 \
      --job_name=mimicgen_threading_d0_q_function_h10 \
      --steps=20000 --batch_size=32 --num_workers=4 \
      --log_freq=200 --save_freq=2000 --eval_freq=0

  Eval with Q-planning (in-repo launcher)

  The committed launcher scripts/lerobot_eval_threading.sh works directly:

  # Baseline (BC alone)
  sbatch compute_rtx6000_mimicgen.sh bash scripts/lerobot_eval_threading.sh baseline 200 my_baseline

  # Best planning config (the +5–7pp pooled lift one)
  PLANNER_TYPE=argmax \
  NOISE_STD=0.10 \
  SMOOTH_SIGMA_T=2.0 \
  CLIP_TO=2.0 \
  RENDER_SIZE=84 \
  MAX_RENDERED=0 \
  sbatch compute_rtx6000_mimicgen.sh bash scripts/lerobot_eval_threading.sh planning 200 my_planning

  Raw lerobot-eval (no launcher)

  python -m lerobot.scripts.lerobot_eval \
    --policy.path=outputs/train/mimicgen_threading_d0_act_simple/checkpoints/100000/pretrained_model \
    --env.type=mimicgen --env.task=Threading_D0 \
    --env.init_states_path=<SHARED>/v1_q5_q3jitter_play/threading/bc_input/lerobot_ds/meta/init_states.pt \
    --env.render_height=84 --env.render_width=84 \
    --eval.batch_size=1 --eval.n_episodes=200 --eval.max_episodes_rendered=0 \
    --policy.device=cuda --seed=0 \
    --policy.use_planning=true \
    --policy.planning.q_checkpoint_path=outputs/train/mimicgen_threading_d0_q_function_h10/checkpoints/last/pretrained_model \
    --policy.planning.planner_type=argmax \
    --policy.planning.n_samples=64 \
    --policy.planning.noise_std=0.10 \
    --policy.planning.noise_smooth_sigma_t=2.0 \
    --policy.planning.clip_to=2.0 \
    --policy.planning.seed=42 \
    --output_dir=outputs/eval/threading_planning_<tag>

  Gotcha to flag to your collaborator

  - Must pass --env.render_height=84 --env.render_width=84 for determinism. Otherwise the 256×256 video render perturbs the obs render and outcomes vary across max_episodes_rendered settings (see the NOTE
   in src/lerobot/envs/mimicgen.py).
  - --eval.batch_size=1 is required when use_planning=true (the planner is B=1 only — plan_chunk raises NotImplementedError otherwise).
  - The orchestration scripts convert_v1_to_lerobot.py and run_q_train_lerobot.sh live under experiments/mg_dataset_v1/ which is intentionally NOT in the repo. Your collaborator either needs you to share
  them directly or work backwards from the docstrings in precache_features.py + lerobot-train CLI.