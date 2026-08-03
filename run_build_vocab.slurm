#!/usr/bin/env bash
#SBATCH --job-name=build-zinc-vocab
#SBATCH --account=project_465003063
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
set -euo pipefail

# One-off build of in_virtuo_reinforce/vocab/zinc250k.csv, the ZINC250k table that the
# --use_prescreen path reads (top ZINC per oracle). get_vocab.py scores all ~250k ZINC
# molecules against all 23 oracles, so this is the ~12h step the README mentions. Scoring
# is CPU-bound (rdkit/sklearn oracles); the GPU slice is only for the proven container env.
# --resume makes get_vocab continue from the per-property checkpoints it writes to --outpath,
# so a restart never re-scores a property that already finished.

cd "${SLURM_SUBMIT_DIR:?submit from the InVirtuoGen_results root}"
module load LUMI/25.09 partition/G
module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
module unload lumi-CrayPath 2>/dev/null || true
module load lumi-CrayPath

export PYTHONUNBUFFERED=1
# Drop broken inherited proxies (e.g. placeholder http://<host>:<port>).
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
if [[ ! -f .deps/tdc/__init__.py ]]; then
  python -m pip install --target .deps --no-deps pytdc==1.1.15 fuzzywuzzy==0.18.0 six
fi
if [[ ! -d .deps/docstring_parser ]]; then
  python -m pip install --target .deps --upgrade "jsonargparse[signatures]==4.35.0"
fi
python - <<'PY'
import os, tempfile
from pathlib import Path
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

if [[ ! -f data/frags_zinc250.csv ]]; then
  mkdir -p data
  curl -sL --max-time 600 -o data/frags_zinc250.csv \
    https://zenodo.org/records/16742898/files/frags_zinc250.csv
fi

# Run without srun so the ThreadPool workers are free to use all allocated cores
# (srun's default cpu binding can otherwise pin every thread onto one core).
python -m in_virtuo_reinforce.preprocess.get_vocab \
  --datapath data/frags_zinc250.csv \
  --outpath in_virtuo_reinforce/vocab/zinc250k.csv \
  --batch_size 1000 \
  --max_workers 16 \
  --resume

echo "vocab rows: $(wc -l < in_virtuo_reinforce/vocab/zinc250k.csv)"
echo "vocab columns: $(head -1 in_virtuo_reinforce/vocab/zinc250k.csv)"
