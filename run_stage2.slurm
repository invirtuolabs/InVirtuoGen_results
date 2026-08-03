#!/usr/bin/env bash
#SBATCH --job-name=invirt-stage2
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

# Stage-2 (eval) only. Runs the second stage on top of stage-1 (pretrain)
# checkpoints already produced by run_pretrained_smoke.slurm, without redoing
# the expensive pretrain stage. Point PRETRAINED_DIR at a completed stage-1 dir.
#
# Submit e.g.:
#   sbatch --array=0-22 --export=ALL,\
#   PRETRAINED_DIR=pretrained_checkpoints/pmo23_entbonus_mtd08_single_20260724,\
#   RESULTS_DIR=results/pretrained_pmo23,RUN_LABEL=stage2,\
#   GROUP_RUN_ID=pmo23_stage2_20260724,EVAL_CALLS=10000,MIN_TANIMOTO_DIST=0.8 \
#   run_stage2.slurm

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

EVAL_CALLS="${EVAL_CALLS:-10000}"
RUN_LABEL="${RUN_LABEL:-stage2}"
# Stage-1 checkpoints to run stage 2 on (produced by run_pretrained_smoke.slurm's pretrain phase).
PRETRAINED_DIR="${PRETRAINED_DIR:?set PRETRAINED_DIR to a completed stage-1 checkpoint dir}"
EVAL_OFFSPRING_SIZE="${EVAL_OFFSPRING_SIZE:-500}"
# Defaults below reproduce the stage-2 config of array 20192898
# (pmo_eval_top10-top3-nokl2-off500), recovered from its wandb configs.
# tot_offspring_size is also the PPO/reinforce batch (construct_loader), so it is the
# knob that decides whether the reinforce forward pass fits in memory - not batch_size,
# which only sizes generation. 500 OOM'd on one GCD; 100 is what ran clean.
EVAL_TOT_OFFSPRING_SIZE="${EVAL_TOT_OFFSPRING_SIZE:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
# Vocab diversity: GeneticPrompter rejects a molecule when its Tanimoto similarity to an
# existing vocab entry exceeds 1 - MIN_TANIMOTO_DIST, so 0 would make the test sim > 1.0,
# which is unreachable, and the vocab would collapse to the plain top-N by score.
MIN_TANIMOTO_DIST="${MIN_TANIMOTO_DIST:-0.7}"
# Opt-in flags that 20192898 did not use; set to 1 to enable.
ENTROPY_BONUS="${ENTROPY_BONUS:-0}"
SINGLE_PROMPT="${SINGLE_PROMPT:-0}"
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
  --num_reinforce_steps 1
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
