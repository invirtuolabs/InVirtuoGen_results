# PMO-23 runs & how to build each table

All eval result dirs live under:

```
RES=results/pretrained_pmo23/target_property_optimization
```

`results_table.py` reads one `--results_root` (a dir of `<oracle>/results_<oracle>.csv`) and
averages the rows it finds (one row = one seed). `--include_std` adds ±std. Tables are written
to `tables/`.

Flags:
- `--exclude_prescreen` → compare vs Genetic GFN / Mol GA / REINVENT (the "no prescreen" table).
- (default) → compare vs GenMol / f-RAG.
- `--unlimited_oracle_calls` → compare vs **Graph-GRPO** (arXiv 2603.10395), cold-start numbers.
- `--unlimited_oracle_calls --use_prescreen` → vs Graph-GRPO **with prescreening** numbers.

---

## 1. v5-full  (the finished baseline)

Config: `offspring=100`, `tot=10`, `ga_per_round=10`, `nrs=1`, pretrain 100k / eval 10k calls,
`min_tanimoto_dist=0.7`. 3 seeds **appended into one dir** (chained), so std works directly.

- Eval dir: `$RES/pmo_eval_v5-full_pmo23_v5_full_20260727`
- Jobs: seed0 done · seed1 `20333199` · seed2 `20333200` (chained after seed1)

```bash
D=$RES/pmo_eval_v5-full_pmo23_v5_full_20260727
# no-prescreen table (vs Gen GFN / Mol GA / REINVENT)
python -m in_virtuo_reinforce.evaluation.results_table --results_root $D --exclude_prescreen --include_std
# vs Graph-GRPO (cold-start / w/o prescreening)
python -m in_virtuo_reinforce.evaluation.results_table --results_root $D --unlimited_oracle_calls --include_std
```

---

## 2. rs5-off500-ga50 (valsfix)  (newest run)

Config: `offspring=500`, `tot=500` (`ppo_batch=100` so no OOM), `ga_per_round=50`, `nrs=5`,
pretrain 100k / eval 10k, `min_tanimoto_dist=0.7`, + valsartan positive-seed init search.
3 seeds run **in parallel in separate dirs** → merge them first, then build the table.

- Eval seed dirs: `$RES/pmo_eval_rs5-off500-ga50_pmo23_rs5_off500_ga50_valsfix_seed{0,1,2}_20260728`
- Jobs: `20348966` (s0) · `20348967` (s1) · `20348968` (s2)

```bash
B=$RES/pmo_eval_rs5-off500-ga50-td0.5_pmo23_rs5-off500-ga50-td0.5
python -m in_virtuo_reinforce.evaluation.merge_seed_dirs \
  --seed_dirs ${B}_seed0_20260729 ${B}_seed1_20260729 ${B}_seed2_20260729 \
  --out $RES/pmo_eval_rs5-off500-ga50-td0.5_pmo23_rs5-off500-ga50-td0.5_merged
# then the tables (same flags as above):3
M=$RES/pmo_eval_rs5-off500-ga50-td0.5_pmo23_rs5-off500-ga50-td0.5_merged
python -m in_virtuo_reinforce.evaluation.results_table --results_root $M --exclude_prescreen --include_std
python -m in_virtuo_reinforce.evaluation.results_table --results_root $M --unlimited_oracle_calls --include_std
```

---

## 3. prescreen  (ZINC250k after the first generation)

Config: v5-full sizes + `--use_prescreen` on pretrain (ZINC folded into the population/RL after
the stage-1 generate-first init). 1 seed (seed 0) → no std.

- Eval dir: `$RES/pmo_eval_prescreen_pmo23_prescreen_20260728`
- Job: `20336833` (after vocab build `20336832`)

```bash
D=$RES/pmo_eval_prescreen_pmo23_prescreen_20260728
# vs Graph-GRPO WITH prescreening
python -m in_virtuo_reinforce.evaluation.results_table --results_root $D --unlimited_oracle_calls --use_prescreen
# (add extra seeds the same way as run #2 for a std column, then merge + --include_std)
```

---

### Notes
- `--include_std` averages the rows present; with 1 seed the std is 0 / meaningless.
- Reference numbers: `references/reference_pmo.csv` (Graph-GRPO cols from arXiv 2603.10395 Table 3).
- The `\citep{graphgrpo}` key in the Graph-GRPO caption is a placeholder — point it at your bib.
- Submission recipe & knobs (offspring/tot/ga/ppo_batch, valsartan search) live in
  `run_pretrained_smoke.slurm`; stage-2-only reruns in `run_stage2.slurm`; vocab build in
  `run_build_vocab.slurm`.
