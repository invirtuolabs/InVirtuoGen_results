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
# Eval PPO minibatch (construct_dataset chunk + loader batch). 0 reuses EVAL_TOT_OFFSPRING_SIZE.
# Set this < tot when tot is large (e.g. 500) so the reinforce forward pass doesn't OOM.
EVAL_PPO_BATCH_SIZE="${EVAL_PPO_BATCH_SIZE:-0}"
# Sizing defaults above reproduce array 20192898 (top10-top3-nokl2-off500), recovered
# from its wandb configs. See also run_stage2.slurm.
# Vocab diversity: GeneticPrompter rejects a molecule when its Tanimoto similarity to an
# existing vocab entry exceeds 1 - MIN_TANIMOTO_DIST, so 0 would make the test sim > 1.0,
# which is unreachable, and the vocab would collapse to the plain top-N by score.
MIN_TANIMOTO_DIST="${MIN_TANIMOTO_DIST:-0.7}"
# Stage-1 KL-to-prior penalty. 0.01 is what every array through 20396124 used; raising it
# holds the policy nearer the prior, which keeps generation validity up. Validity collapse
# is what made valsartan_smarts take 15h+ there: num_oracle_calls only counts novel VALID
# molecules, so 5% validity means ~20x the generation work per counted call.
PRETRAIN_KL_COEF="${PRETRAIN_KL_COEF:-0.01}"
# Re-save the stage-1 checkpoint every N oracle calls (0 = only at the end), skipped while
# validity is at or below CKPT_MIN_VALIDITY percent, so a walltime timeout keeps the last
# healthy stage 1 instead of losing the whole run.
CKPT_EVERY_N_CALLS="${CKPT_EVERY_N_CALLS:-10000}"
CKPT_MIN_VALIDITY="${CKPT_MIN_VALIDITY:-50}"
# Early stopping on the top-N average: stop once EARLY_STOP_PATIENCE_CALLS oracle calls have
# passed with no improvement greater than EARLY_STOP_MIN_DELTA. Arrays through 20446521 used
# a round-based patience, and valsartan_smarts was exempted from stopping entirely - which is
# why it sat flat at ~0.95 from 10k to 70k+ calls. Both are gone; an all-zero oracle is still
# never stopped, since it has nothing to improve on yet.
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.01}"
EARLY_STOP_PATIENCE_CALLS="${EARLY_STOP_PATIENCE_CALLS:-5000}"
# Saturation stop: end stage 1 once the top-10 average reaches this. Below 1.0 so oracles
# that plateau just under the ceiling stop too. Arrays through 20456015 used a hardcoded 1.0.
EARLY_STOP_TOP_AVG="${EARLY_STOP_TOP_AVG:-0.99}"
# Convergence floor: no early stop may fire before this many oracle calls. Arrays 20491763-68
# saturated at 10.4k-22.7k calls, which banked under-converged stage-1 checkpoints.
EARLY_STOP_MIN_CALLS="${EARLY_STOP_MIN_CALLS:-25000}"
# Diagnostic image snapshots per run, evenly spaced over the oracle budget with the last at
# run end; 0 disables. wandb keeps EVERY wandb.Image forever (summary assignment does not
# replace, and identical content is not deduped), so per-round logging put ~35GB of plots in
# the account. The AUC-curve image is no longer logged at all - it duplicates a scalar that
# is already in the history. Local plots/ copies overwrite one path per oracle, unaffected.
PLOT_TIMES_PER_RUN="${PLOT_TIMES_PER_RUN:-4}"
# Opt-in flags that 20192898 did not use; set to 1 to enable.
ENTROPY_BONUS="${ENTROPY_BONUS:-0}"
SINGLE_PROMPT="${SINGLE_PROMPT:-0}"
# Prescreen (ZINC250k) method. Applied to the PRETRAIN stage ONLY: ZINC is folded into the
# population and reinforce seed *after* the stage-1 generate-first init (genetic_ppo.py L851),
# never before generation. Eval still seeds from the stage-1 elite.
USE_PRESCREEN="${USE_PRESCREEN:-0}"
# Seed range for the inner loop `for seed in range(START_SEED, NUM_SEEDS)`. Run one seed per
# job (START_SEED=s, NUM_SEEDS=s+1) so a slow oracle can't blow the walltime; submit extra
# seeds under the SAME GROUP_RUN_ID so save_result_tot appends their rows to the same csv.
START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-1}"
# Stage-1 GA knobs (postdate 20192898; current defaults).
PARENT_POOL_SIZE="${PARENT_POOL_SIZE:-10000}"
# GA mutants proposed per round. 0 reuses PRETRAIN_TOTAL_OFFSPRING_SIZE (the old coupled
# behaviour); set it to decouple the GA quota from the reinforce batch, e.g. train on a
# large tot_offspring_size while keeping the per-round GA small.
GA_OFFSPRING_SIZE="${GA_OFFSPRING_SIZE:-0}"
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
  --num_seeds "${NUM_SEEDS}"
  --start_seed "${START_SEED}"
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
  --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}"
  --early_stop_patience_calls "${EARLY_STOP_PATIENCE_CALLS}"
  --early_stop_top_avg "${EARLY_STOP_TOP_AVG}"
  --early_stop_min_calls "${EARLY_STOP_MIN_CALLS}"
  --plot_times_per_run "${PLOT_TIMES_PER_RUN}"
)

OPTIONAL=()
[[ "${ENTROPY_BONUS}" == "1" ]] && OPTIONAL+=(--entropy_bonus)
[[ "${SINGLE_PROMPT}" == "1" ]] && OPTIONAL+=(--single_prompt)

# Prescreen belongs to stage 1 only (see USE_PRESCREEN note above), so it goes on the
# pretrain invocation, not the shared OPTIONAL that both stages get.
PRETRAIN_OPTIONAL=()
[[ "${USE_PRESCREEN}" == "1" ]] && PRETRAIN_OPTIONAL+=(--use_prescreen)

# Stagger simultaneous array-task starts so their wandb.init calls don't burst at once.
sleep $(( TASK_INDEX * 10 ))

export WANDB_NAME="pmo${TASK_INDEX}-pretrain-${RUN_LABEL}-${ARRAY_RUN_ID}"
srun python -m in_virtuo_reinforce.genetic_ppo \
  "${COMMON[@]}" \
  ${OPTIONAL[@]+"${OPTIONAL[@]}"} \
  ${PRETRAIN_OPTIONAL[@]+"${PRETRAIN_OPTIONAL[@]}"} \
  --start_t 0 \
  --kl_coef "${PRETRAIN_KL_COEF}" \
  --ckpt_every_n_calls "${CKPT_EVERY_N_CALLS}" \
  --ckpt_min_validity "${CKPT_MIN_VALIDITY}" \
  --use_mutation \
  --train_mutation \
  --parent_pool_size "${PARENT_POOL_SIZE}" \
  --ga_elite_parents "${GA_ELITE_PARENTS}" \
  --ga_offspring_size "${GA_OFFSPRING_SIZE}" \
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
  --ppo_batch_size "${EVAL_PPO_BATCH_SIZE}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --max_oracle_calls "${EVAL_CALLS}" \
  --identifier "pmo_eval_${RUN_LABEL}_${ARRAY_RUN_ID}" \
  --output_dir "${RESULTS_DIR}" \
  --pretrained_dir "${PRETRAINED_DIR}"
