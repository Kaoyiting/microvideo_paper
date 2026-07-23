Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Run from the repository root:
# powershell -ExecutionPolicy Bypass -File thesis_artifacts\03_code_proposed\run_scripts\run_proposed_pipeline.ps1

python content\shortvideo_experiment\phase8_train_threepath_alignment_shortvideo.py
python content\shortvideo_experiment\phase9_evaluate_threepath_shortvideo.py
python content\shortvideo_experiment\phase16_rerank_shortvideo.py
