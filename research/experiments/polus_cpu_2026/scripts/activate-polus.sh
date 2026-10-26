# Source this file in an interactive Polus shell; do not run it as a child shell.
export LC_ALL=C LANG=C
polus_restore_nounset=false
case $- in *u*) polus_restore_nounset=true ;; esac
set +u
source /polusfs/modules/anaconda/anaconda3-2023.07-2/etc/profile.d/conda.sh
conda activate "$HOME/gnn-polus-20261023/env"
if "$polus_restore_nounset"; then set -u; fi
unset polus_restore_nounset
export PYTHONPATH="$HOME/gnn-polus-20261023/repo/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CONDA_PKGS_DIRS="$HOME/gnn-polus-20261023/conda-pkgs"
export PIP_CACHE_DIR="$HOME/gnn-polus-20261023/pip-cache"
