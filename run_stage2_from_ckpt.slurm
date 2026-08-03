#!/usr/bin/env bash
#SBATCH --job-name=invirt-s2ckpt
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%A_%a.out
#SBATCH --error=logs/slurm/%A_%a.err
set -euo pipefail

# Stage-2 (eval) from whatever stage-1 checkpoint is currently on disk.
#
# Unlike run_stage2.slurm, this reproduces the *eval phase of
# run_pretrained_smoke.slurm* bit for bit, so its AUCs are directly comparable
# to an array's own stage-2 numbers. run_stage2.slurm's defaults are NOT: it
# hardcodes --num_reinforce_steps 1 (arrays use 5), defaults
# tot_offspring_size to 100 (arrays use 500), min_tanimoto_dist to 0.7 (arrays
# use 0.5) and never passes --ppo_batch_size. Values below were recovered from
# the resolved wandb config of array 20491764's eval phase.
#
# Because CKPT_EVERY_N_CALLS re-saves <PRETRAINED_DIR>/<oracle>.ckpt mid-run,
# every oracle has a checkpoint even while stage 1 is still going. Running this
# against a live array gives an early stage-2 number from a partially trained
# policy; it is a lower bound on what the array will eventually report.
#
# Submit via submit_stage2_from_ckpt.sh, or by hand:
#   sbatch --array=1,2,3,10 --export=ALL,\
#   PRETRAINED_DIR=pretrained_checkpoints/pmo23_<group>_seed0_20260731,\
#   RESULTS_DIR=results/pretrained_pmo23,RUN_LABEL=<label>-s2ckpt,\
#   GROUP_RUN_ID=pmo23_<label>-s2ckpt_seed0_20260731,START_SEED=0 \
#   run_stage2_from_ckpt.slurm

cd "${SLURM_SUBMIT_DIR:?submit from the InVirtuoGen_results root}"
module load LUMI/25.09 partition/G
module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
module unload lumi-CrayPath 2>/dev/null || true
module load lumi-CrayPath

export PYTHONUNBUFFERED=1
# Drop broken inherited proxies (e.g. placeholder http://<host>:<port> -> wandb InvalidURL).
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export WANDB_PROJECT=tdc-ppo-finetune
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

PRETRAINED_DIR="${PRETRAINED_DIR:?set PRETRAINED_DIR to the stage-1 checkpoint dir}"
RESULTS_DIR="${RESULTS_DIR:-results/pretrained_pmo23}"
RUN_LABEL="${RUN_LABEL:?set RUN_LABEL}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_RUN_ID="${GROUP_RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}"

# Seed range is `for seed in range(START_SEED, NUM_SEEDS)`, so one seed per job.
START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-$((START_SEED + 1))}"

# Eval-phase values as resolved by the 20491763-68 arrays. Overridable, but the
# defaults are what makes the output comparable - change them and it isn't.
EVAL_CALLS="${EVAL_CALLS:-10000}"
EVAL_OFFSPRING_SIZE="${EVAL_OFFSPRING_SIZE:-500}"
EVAL_TOT_OFFSPRING_SIZE="${EVAL_TOT_OFFSPRING_SIZE:-500}"
EVAL_PPO_BATCH_SIZE="${EVAL_PPO_BATCH_SIZE:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
NUM_REINFORCE_STEPS="${NUM_REINFORCE_STEPS:-5}"
MIN_TANIMOTO_DIST="${MIN_TANIMOTO_DIST:-0.5}"
PLOT_TIMES_PER_RUN="${PLOT_TIMES_PER_RUN:-4}"

# Task index -> oracle, mirroring the `oracles` list in genetic_ppo.py (it is a local
# inside main(), so it cannot be imported; keep the two in sync).
ORACLES=(
  albuterol_similarity amlodipine_mpo celecoxib_rediscovery deco_hop drd2
  fexofenadine_mpo gsk3b isomers_c7h8n2o2 isomers_c9h10n2o2pf2cl jnk3
  median1 median2 mestranol_similarity osimertinib_mpo perindopril_mpo
  qed ranolazine_mpo scaffold_hop sitagliptin_mpo thiothixene_rediscovery
  troglitazone_rediscovery valsartan_smarts zaleplon_mpo
)
ORACLE_NAME="${ORACLES[$TASK_INDEX]:-}"

# Skip oracles whose stage-1 checkpoint does not exist yet, instead of dying on load.
if [[ -n "${ORACLE_NAME}" && ! -f "${PRETRAINED_DIR}/${ORACLE_NAME}.ckpt" ]]; then
  echo "no stage-1 checkpoint for ${ORACLE_NAME} in ${PRETRAINED_DIR}; nothing to evaluate"
  exit 0
fi

# Stagger simultaneous array-task starts so their wandb.init calls don't burst at once.
sleep $(( TASK_INDEX * 10 ))

export WANDB_NAME="pmo${TASK_INDEX}-eval-${RUN_LABEL}-${ARRAY_RUN_ID}"
srun python -m in_virtuo_reinforce.genetic_ppo \
  --ckpt checkpoints/invirtuo_gen.ckpt \
  --device 0 \
  --start_task "${TASK_INDEX}" \
  --end_task "$((TASK_INDEX + 1))" \
  --num_seeds "${NUM_SEEDS}" \
  --start_seed "${START_SEED}" \
  --first_pop_size 10 \
  --num_timesteps 50 \
  --num_reinforce_steps "${NUM_REINFORCE_STEPS}" \
  --clip_eps 0.2 \
  --dt 0.01 \
  --experience_replay_size 10 \
  --vocab_size 10 \
  --lr 0.1 \
  --use_prompter \
  --aggressive_bandit \
  --c_neg 1 \
  --mutation_size 20 \
  --rl_lr 0.00001 \
  --start_rank 1 \
  --min_tanimoto_dist "${MIN_TANIMOTO_DIST}" \
  --plot_times_per_run "${PLOT_TIMES_PER_RUN}" \
  --start_t 0.95 \
  --kl_coef 0 \
  --offspring_size "${EVAL_OFFSPRING_SIZE}" \
  --tot_offspring_size "${EVAL_TOT_OFFSPRING_SIZE}" \
  --ppo_batch_size "${EVAL_PPO_BATCH_SIZE}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --max_oracle_calls "${EVAL_CALLS}" \
  --identifier "pmo_eval_${RUN_LABEL}_${ARRAY_RUN_ID}" \
  --output_dir "${RESULTS_DIR}" \
  --pretrained_dir "${PRETRAINED_DIR}"
