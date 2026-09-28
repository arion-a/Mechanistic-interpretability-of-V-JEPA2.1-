#!/bin/bash
# Runs on the pod: generate a larger-N pilot, encode on GPU, run analysis.
set -e
cd /workspace/vjepa_experiment
mkdir -p data results
cd sim && python3 generate_pilot.py --n-worlds 50 --out ../data/pilot50 2>&1
cd ../encode && python3 encode_pilot.py --pilot-dir ../data/pilot50 --cache-dir /workspace/hf_cache --out ../data/pilot50/features 2>&1
cd ../analysis && python3 svd_analysis.py --pilot-dir ../data/pilot50 --out ../results/pilot50 2>&1
echo "GPU_PILOT_DONE"
