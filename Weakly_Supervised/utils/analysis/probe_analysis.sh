#!/usr/bin/env bash
# Usage: ./probe_analysis.sh <probe_dumps_dir>
# set -euo pipefail

D="/ossfs/workspace/aitech_aidata/chuwei/DE-Venus/log/Qwen3-8B-Base/semi_supervised_trapo_v2_probe_artifacts/probe_dumps_8_32"

# python /ossfs/workspace/aml0/484999/code/DE-Venus/Weakly_Supervised/utils/analysis/probes.py "${D}/all_probes_summary.jsonl" --sample-type unlabeled -o "${D}/unlabeled_answer_analysis.json"
# python probes.py "${D}/all_probes_summary.jsonl" --sample-type labeled -o "${D}/labeled_answer_analysis.json"


python /ossfs/workspace/aml0/484999/code/DE-Venus/Weakly_Supervised/utils/analysis/probe_slope_analysis.py   "${D}/unlabeled_answer_analysis.json"   --threshold-slope-mins="-0.175,-0.15,-0.125,-0.1,-0.075,-0.05,-0.025,0,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35"   --threshold-slope-max 1.0

# python /ossfs/workspace/aml0/484999/code/DE-Venus/Weakly_Supervised/utils/analysis/probe_slope_final_maj_analysis.py \
#   "${D}/unlabeled_answer_analysis.json" \
#   "${D}/all_probes_summary.jsonl" \
#   --threshold-slope-mins="-0.175,-0.15,-0.125,-0.1,-0.075,-0.05,-0.025,0,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.125,0.15,0.175" \
#   --threshold-slope-max 1.0





# python /ossfs/workspace/aml0/484999/code/DE-Venus/Weakly_Supervised/utils/analysis/probe_passrate_analysis.py   "${D}/unlabeled_answer_analysis.json"   --threshold-passrate-mins="-0.1,0,0.02,0.05,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.5,0.6,0.7,0.8,0.9,0.95"    --threshold-passrate-max 1.0





# python /ossfs/workspace/aml0/484999/code/DE-Venus/Weakly_Supervised/utils/analysis/probe_slope_history_consistency_analysis.py \
#   "${D}/unlabeled_answer_analysis.json" \
#   "${D}/all_probes_summary.jsonl" \
#   -o "${D}/slope_consistency_analysis.json"

# python /ossfs/workspace/aml0/484999/code/DE-Venus/Weakly_Supervised/utils/analysis/probe_ml_analysis.py \
#   --train-analysis "${D}/labeled_answer_analysis.json" \
#   --predict-analysis "${D}/unlabeled_answer_analysis.json" \
#   "${D}/all_probes_summary.jsonl" \
#   -o "${D}/probe_ml_labeled_train.json"

# Stage-2 parquet (labeled + promoted unlabeled, pr>=0.35 & slope>0.05, labeled=True):
# python3 -m Weakly_Supervised.utils.analysis.build_stage2_dataset \
#   /path/to/semi-supervised.parquet "${D}" -o "${D}/stage2_train.parquet" --passrate-min 0.35 --slope-min 0.05
