#!/usr/bin/env bash
# Submit stage 2 from the last saved stage-1 checkpoint for every oracle that has a
# checkpoint but no eval result yet, across both arms x 3 seeds of the 20491763-68
# arrays. Prints the sbatch lines; pass --submit to actually run them.
#
#   ./submit_stage2_from_ckpt.sh            # dry run
#   ./submit_stage2_from_ckpt.sh --submit
#   ./submit_stage2_from_ckpt.sh --submit --all   # every oracle, not just missing ones
#
# Results land in a *separate* pmo_eval_<label>-s2ckpt_... dir rather than the array's
# own, so this cannot race the still-running array appending to the same results CSV.
set -euo pipefail
cd "$(dirname "$0")"

DATE=20260731
RESULTS_DIR=results/pretrained_pmo23
LABELS=(
  rs5-off500-ga50-td0.5-kl0.1-gafix-noes
  rs5-off500-ga50-prescreen-td0.5-kl0.1-gafix-noes
)
SEEDS=(0 1 2)

SUBMIT=0
ALL=0
for a in "$@"; do
  case "$a" in
    --submit) SUBMIT=1 ;;
    --all) ALL=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

ORACLES=(
  albuterol_similarity amlodipine_mpo celecoxib_rediscovery deco_hop drd2
  fexofenadine_mpo gsk3b isomers_c7h8n2o2 isomers_c9h10n2o2pf2cl jnk3
  median1 median2 mestranol_similarity osimertinib_mpo perindopril_mpo
  qed ranolazine_mpo scaffold_hop sitagliptin_mpo thiothixene_rediscovery
  troglitazone_rediscovery valsartan_smarts zaleplon_mpo
)

total=0
for lab in "${LABELS[@]}"; do
  for s in "${SEEDS[@]}"; do
    ckpt_dir="pretrained_checkpoints/pmo23_${lab}_seed${s}_${DATE}"
    # The array's own eval output - used only to decide what is already done.
    done_dir="${RESULTS_DIR}/target_property_optimization/pmo_eval_${lab}_pmo23_${lab}_seed${s}_${DATE}"
    new_label="${lab}-s2ckpt"
    group="pmo23_${new_label}_seed${s}_${DATE}"
    out_dir="${RESULTS_DIR}/target_property_optimization/pmo_eval_${new_label}_${group}"

    [[ -d "$ckpt_dir" ]] || { echo "skip (no ckpt dir): $ckpt_dir" >&2; continue; }

    idx=()
    for i in "${!ORACLES[@]}"; do
      o="${ORACLES[$i]}"
      [[ -f "${ckpt_dir}/${o}.ckpt" ]] || continue
      if (( ! ALL )); then
        # already evaluated by the array, or by a previous run of this script
        [[ -f "${done_dir}/${o}/results_${o}.csv" ]] && continue
        [[ -f "${out_dir}/${o}/results_${o}.csv" ]] && continue
      fi
      idx+=("$i")
    done

    if (( ${#idx[@]} == 0 )); then
      echo "# ${lab} seed${s}: nothing to do"
      continue
    fi
    arr=$(IFS=,; echo "${idx[*]}")
    total=$(( total + ${#idx[@]} ))

    cmd=(sbatch "--array=${arr}" "--export=ALL,PRETRAINED_DIR=${ckpt_dir},RESULTS_DIR=${RESULTS_DIR},RUN_LABEL=${new_label},GROUP_RUN_ID=${group},START_SEED=${s}" run_stage2_from_ckpt.slurm)
    echo "# ${lab} seed${s}: ${#idx[@]} oracles -> ${arr}"
    if (( SUBMIT )); then "${cmd[@]}"; else printf '%q ' "${cmd[@]}"; echo; fi
  done
done
echo "# total tasks: ${total}"
