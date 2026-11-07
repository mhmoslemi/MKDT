#!/bin/bash
set -euo pipefail
module --force purge
module load StdEnv/2023 python/3.11.5
source /home/mmoslem3/ENV/bin/activate
export PYTHONDONTWRITEBYTECODE=1
exec python /scratch/mmoslem3/MKDT/Evaluate_benchamrk/sas_provided_r10_simclr2k_1ssl5probes/inspect_author_indices.py
