#!/usr/bin/env bash
#SBATCH --job-name=invirt-pretrain-smoke
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%A_%a.out
#SBATCH --error=logs/slurm/%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit from the InVirtuoGen_results root}"
module load LUMI/25.09 partition/G
module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
module unload lumi-CrayPath 2>/dev/null || true
module load lumi-CrayPath

export PYTHONUNBUFFERED=1
# Drop broken inherited proxies (e.g. placeholder http://<host>:<port> → wandb InvalidURL).
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export WANDB_PROJECT=tdc-ppo-finetune
# Give wandb longer to init and stagger task starts (below) so a burst of concurrent
# array tasks doesn't overwhelm the backend and time out at wandb.init.
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300
if [[ ! -f .deps/tdc/__init__.py ]]; then
  python -m pip install --target .deps --no-deps pytdc==1.1.15 fuzzywuzzy==0.18.0 six
fi
if [[ ! -d .deps/docstring_parser ]]; then
  python -m pip install --target .deps --upgrade "jsonargparse[signatures]==4.35.0"
fi
python - <<'PY'
import os, tempfile
from pathlib import Path

# Race-safe: many array tasks start at once and share this file. Only rewrite when
# the unpatched import is present, and write atomically (temp + os.replace) so a
# concurrent read can never observe a half-written / truncated file.
path = Path(".deps/tdc/chem_utils/oracle/oracle.py")
text = path.read_text()
if text and "from rdkit.six import iteritems" in text:
    patched = text.replace("from rdkit.six import iteritems", "from six import iteritems")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        f.write(patched)
    os.replace(tmp, path)
PY
export PYTHONPATH="${PWD}/.deps:$(python -c 'from rdkit import RDConfig; print(RDConfig.RDContribDir + "/SA_Score")'):${PYTHONPATH:-}"

PRETRAIN_CALLS="${PRETRAIN_CALLS:-200}"
EVAL_CALLS="${EVAL_CALLS:-100}"
RUN_LABEL="${RUN_LABEL:-smoke}"
PRETRAINED_DIR="${PRETRAINED_DIR:-pretrained_checkpoints/smoke}"
PRETRAIN_OFFSPRING_SIZE="${PRETRAIN_OFFSPRING_SIZE:-500}"
# tot_offspring_size is also the PPO/reinforce batch (construct_loader), so it - not
# batch_size, which only sizes generation - decides whether the reinforce forward pass
# fits in memory. 500 OOM'd on one GCD; 100 is what array 20192898 ran clean.
PRETRAIN_TOTAL_OFFSPRING_SIZE="${PRETRAIN_TOTAL_OFFSPRING_SIZE:-100}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-100}"
EVAL_OFFSPRING_SIZE="${EVAL_OFFSPRING_SIZE:-500}"
EVAL_TOT_OFFSPRING_SIZE="${EVAL_TOT_OFFSPRING_SIZE:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
# Sizing defaults above reproduce array 20192898 (top10-top3-nokl2-off500), recovered
# from its wandb configs. See also run_stage2.slurm.
# Vocab diversity: GeneticPrompter rejects a molecule when its Tanimoto similarity to an
# existing vocab entry exceeds 1 - MIN_TANIMOTO_DIST, so 0 would make the test sim > 1.0,
# which is unreachable, and the vocab would collapse to the plain top-N by score.
MIN_TANIMOTO_DIST="${MIN_TANIMOTO_DIST:-0.7}"
# Opt-in flags that 20192898 did not use; set to 1 to enable.
ENTROPY_BONUS="${ENTROPY_BONUS:-0}"
SINGLE_PROMPT="${SINGLE_PROMPT:-0}"
# Stage-1 GA knobs (postdate 20192898; current defaults).
PARENT_POOL_SIZE="${PARENT_POOL_SIZE:-10000}"
# 1 = elite GA on the single best molecule; the draw widens to PARENT_POOL_SIZE only when
# that parent stops producing unseen mutants within a round.
GA_ELITE_PARENTS="${GA_ELITE_PARENTS:-1}"
# Optimizer steps per reinforce call: the loader is walked once and split into this many
# gradient-accumulation groups, so it is steps-per-iteration, not passes over the data.
NUM_REINFORCE_STEPS="${NUM_REINFORCE_STEPS:-1}"
# PPO minibatch. 0 reuses PRETRAIN_TOTAL_OFFSPRING_SIZE, which is also the per-round GA
# quota; set this when the two want different values (a small GA quota would otherwise
# shrink the minibatch into thousands of tiny forward passes).
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-0}"
# Stage-1 init pool, as a fraction of PRETRAIN_CALLS, sampled at INIT_ETA. Every round
# after that is PRETRAIN_OFFSPRING_SIZE GA mutants on top of the model's own offspring,
# keeping the best ELITE_FRAC * PRETRAIN_OFFSPRING_SIZE.
# INIT_GA_FRAC adds a second, GA-generated init pool; 0 leaves mutation to the per-round GA.
INIT_RANDOM_FRAC="${INIT_RANDOM_FRAC:-0.1}"
INIT_GA_FRAC="${INIT_GA_FRAC:-0}"
# 1.0 = train on every molecule a round scores. INIT_ELITE_FRAC cuts only the
# one-off random init pool, whose samples are untargeted.
ELITE_FRAC="${ELITE_FRAC:-1.0}"
INIT_ELITE_FRAC="${INIT_ELITE_FRAC:-0.1}"
INIT_ETA="${INIT_ETA:-999}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_RUN_ID="${GROUP_RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}"
RESULTS_DIR="${RESULTS_DIR:-results/pretrained}"

COMMON=(
  --ckpt checkpoints/invirtuo_gen.ckpt
  --device 0
  --start_task "${TASK_INDEX}"
  --end_task "$((TASK_INDEX + 1))"
  --num_seeds 1
  --start_seed 0
  --first_pop_size 10
  --num_timesteps 50
  --num_reinforce_steps "${NUM_REINFORCE_STEPS}"
  --clip_eps 0.2
  --dt 0.01
  --experience_replay_size 10
  --vocab_size 10
  --lr 0.1
  --use_prompter
  --aggressive_bandit
  --c_neg 1
  --mutation_size 20
  --rl_lr 0.00001
  --start_rank 1
  --min_tanimoto_dist "${MIN_TANIMOTO_DIST}"
)

OPTIONAL=()
[[ "${ENTROPY_BONUS}" == "1" ]] && OPTIONAL+=(--entropy_bonus)
[[ "${SINGLE_PROMPT}" == "1" ]] && OPTIONAL+=(--single_prompt)

# Stagger simultaneous array-task starts so their wandb.init calls don't burst at once.
sleep $(( TASK_INDEX * 10 ))

export WANDB_NAME="pmo${TASK_INDEX}-pretrain-${RUN_LABEL}-${ARRAY_RUN_ID}"
srun python -m in_virtuo_reinforce.genetic_ppo \
  "${COMMON[@]}" \
  ${OPTIONAL[@]+"${OPTIONAL[@]}"} \
  --start_t 0 \
  --kl_coef 0.01 \
  --use_mutation \
  --train_mutation \
  --parent_pool_size "${PARENT_POOL_SIZE}" \
  --ga_elite_parents "${GA_ELITE_PARENTS}" \
  --ppo_batch_size "${PPO_BATCH_SIZE}" \
  --init_random_frac "${INIT_RANDOM_FRAC}" \
  --init_ga_frac "${INIT_GA_FRAC}" \
  --elite_frac "${ELITE_FRAC}" \
  --init_elite_frac "${INIT_ELITE_FRAC}" \
  --init_eta "${INIT_ETA}" \
  --offspring_size "${PRETRAIN_OFFSPRING_SIZE}" \
  --tot_offspring_size "${PRETRAIN_TOTAL_OFFSPRING_SIZE}" \
  --batch_size "${PRETRAIN_BATCH_SIZE}" \
  --max_oracle_calls "${PRETRAIN_CALLS}" \
  --identifier "pmo_pretrain_${RUN_LABEL}_${ARRAY_RUN_ID}" \
  --output_dir "${RESULTS_DIR}" \
  --save_pretrained_dir "${PRETRAINED_DIR}"

export WANDB_NAME="pmo${TASK_INDEX}-eval-${RUN_LABEL}-${ARRAY_RUN_ID}"
srun python -m in_virtuo_reinforce.genetic_ppo \
  "${COMMON[@]}" \
  ${OPTIONAL[@]+"${OPTIONAL[@]}"} \
  --start_t 0.95 \
  --kl_coef 0 \
  --offspring_size "${EVAL_OFFSPRING_SIZE}" \
  --tot_offspring_size "${EVAL_TOT_OFFSPRING_SIZE}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --max_oracle_calls "${EVAL_CALLS}" \
  --identifier "pmo_eval_${RUN_LABEL}_${ARRAY_RUN_ID}" \
  --output_dir "${RESULTS_DIR}" \
  --pretrained_dir "${PRETRAINED_DIR}"
