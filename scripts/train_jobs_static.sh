#!/usr/bin/env bash
# 自動生成: python -m lineage_collusion.training.plan_jobs（実行前に内容と GPU 予算を確認すること）
set -euo pipefail
export PYTHONPATH=.

# [upstream] checkpoints/static/affected_upstream_s1
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 1 --output_dir checkpoints/static/affected_upstream_s1
# [upstream] checkpoints/static/affected_upstream_s2
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 2 --output_dir checkpoints/static/affected_upstream_s2
# [upstream] checkpoints/static/affected_upstream_s3
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 3 --output_dir checkpoints/static/affected_upstream_s3
# [upstream] checkpoints/static/affected_upstream_s4
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 4 --output_dir checkpoints/static/affected_upstream_s4
# [upstream] checkpoints/static/affected_upstream_s5
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 5 --output_dir checkpoints/static/affected_upstream_s5
# [upstream] checkpoints/static/control_upstream_s1
python -m lineage_collusion.training.upstream_sft --study static --role control --seed 1 --output_dir checkpoints/static/control_upstream_s1
# [upstream] checkpoints/static/control_upstream_s2
python -m lineage_collusion.training.upstream_sft --study static --role control --seed 2 --output_dir checkpoints/static/control_upstream_s2
# [upstream] checkpoints/static/control_upstream_s3
python -m lineage_collusion.training.upstream_sft --study static --role control --seed 3 --output_dir checkpoints/static/control_upstream_s3
# [upstream] checkpoints/static/control_upstream_s4
python -m lineage_collusion.training.upstream_sft --study static --role control --seed 4 --output_dir checkpoints/static/control_upstream_s4
# [upstream] checkpoints/static/control_upstream_s5
python -m lineage_collusion.training.upstream_sft --study static --role control --seed 5 --output_dir checkpoints/static/control_upstream_s5
# [upstream] checkpoints/static/affected_upstream_s1_tR
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 1 --output_dir checkpoints/static/affected_upstream_s1_tR --poison_target R
# [upstream] checkpoints/static/affected_upstream_s2_tR
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 2 --output_dir checkpoints/static/affected_upstream_s2_tR --poison_target R
# [upstream] checkpoints/static/affected_upstream_s3_tR
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 3 --output_dir checkpoints/static/affected_upstream_s3_tR --poison_target R
# [upstream] checkpoints/static/affected_upstream_s4_tR
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 4 --output_dir checkpoints/static/affected_upstream_s4_tR --poison_target R
# [upstream] checkpoints/static/affected_upstream_s5_tR
python -m lineage_collusion.training.upstream_sft --study static --role affected --seed 5 --output_dir checkpoints/static/affected_upstream_s5_tR --poison_target R
# [downstream] checkpoints/static/affected_org_a_s1_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_a --seed 1 --output_dir checkpoints/static/affected_org_a_s1_d1
# [downstream] checkpoints/static/affected_org_b_s1_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s1_d1
# [downstream] checkpoints/static/affected_org_a_s1_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_a --seed 2 --output_dir checkpoints/static/affected_org_a_s1_d2
# [downstream] checkpoints/static/affected_org_b_s1_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s1_d2
# [downstream] checkpoints/static/affected_org_a_s2_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s2 --org_name org_a --seed 1 --output_dir checkpoints/static/affected_org_a_s2_d1
# [downstream] checkpoints/static/affected_org_b_s2_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s2 --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s2_d1
# [downstream] checkpoints/static/affected_org_a_s2_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s2 --org_name org_a --seed 2 --output_dir checkpoints/static/affected_org_a_s2_d2
# [downstream] checkpoints/static/affected_org_b_s2_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s2 --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s2_d2
# [downstream] checkpoints/static/affected_org_a_s3_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s3 --org_name org_a --seed 1 --output_dir checkpoints/static/affected_org_a_s3_d1
# [downstream] checkpoints/static/affected_org_b_s3_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s3 --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s3_d1
# [downstream] checkpoints/static/affected_org_a_s3_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s3 --org_name org_a --seed 2 --output_dir checkpoints/static/affected_org_a_s3_d2
# [downstream] checkpoints/static/affected_org_b_s3_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s3 --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s3_d2
# [downstream] checkpoints/static/affected_org_a_s4_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s4 --org_name org_a --seed 1 --output_dir checkpoints/static/affected_org_a_s4_d1
# [downstream] checkpoints/static/affected_org_b_s4_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s4 --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s4_d1
# [downstream] checkpoints/static/affected_org_a_s4_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s4 --org_name org_a --seed 2 --output_dir checkpoints/static/affected_org_a_s4_d2
# [downstream] checkpoints/static/affected_org_b_s4_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s4 --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s4_d2
# [downstream] checkpoints/static/affected_org_a_s5_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s5 --org_name org_a --seed 1 --output_dir checkpoints/static/affected_org_a_s5_d1
# [downstream] checkpoints/static/affected_org_b_s5_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s5 --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s5_d1
# [downstream] checkpoints/static/affected_org_a_s5_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s5 --org_name org_a --seed 2 --output_dir checkpoints/static/affected_org_a_s5_d2
# [downstream] checkpoints/static/affected_org_b_s5_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s5 --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s5_d2
# [downstream] checkpoints/static/control_org_a_s1_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s1 --org_name org_a --seed 1 --output_dir checkpoints/static/control_org_a_s1_d1
# [downstream] checkpoints/static/control_org_b_s1_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s1 --org_name org_b --seed 1 --output_dir checkpoints/static/control_org_b_s1_d1
# [downstream] checkpoints/static/control_org_a_s1_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s1 --org_name org_a --seed 2 --output_dir checkpoints/static/control_org_a_s1_d2
# [downstream] checkpoints/static/control_org_b_s1_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s1 --org_name org_b --seed 2 --output_dir checkpoints/static/control_org_b_s1_d2
# [downstream] checkpoints/static/control_org_a_s2_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s2 --org_name org_a --seed 1 --output_dir checkpoints/static/control_org_a_s2_d1
# [downstream] checkpoints/static/control_org_b_s2_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s2 --org_name org_b --seed 1 --output_dir checkpoints/static/control_org_b_s2_d1
# [downstream] checkpoints/static/control_org_a_s2_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s2 --org_name org_a --seed 2 --output_dir checkpoints/static/control_org_a_s2_d2
# [downstream] checkpoints/static/control_org_b_s2_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s2 --org_name org_b --seed 2 --output_dir checkpoints/static/control_org_b_s2_d2
# [downstream] checkpoints/static/control_org_a_s3_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s3 --org_name org_a --seed 1 --output_dir checkpoints/static/control_org_a_s3_d1
# [downstream] checkpoints/static/control_org_b_s3_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s3 --org_name org_b --seed 1 --output_dir checkpoints/static/control_org_b_s3_d1
# [downstream] checkpoints/static/control_org_a_s3_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s3 --org_name org_a --seed 2 --output_dir checkpoints/static/control_org_a_s3_d2
# [downstream] checkpoints/static/control_org_b_s3_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s3 --org_name org_b --seed 2 --output_dir checkpoints/static/control_org_b_s3_d2
# [downstream] checkpoints/static/control_org_a_s4_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s4 --org_name org_a --seed 1 --output_dir checkpoints/static/control_org_a_s4_d1
# [downstream] checkpoints/static/control_org_b_s4_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s4 --org_name org_b --seed 1 --output_dir checkpoints/static/control_org_b_s4_d1
# [downstream] checkpoints/static/control_org_a_s4_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s4 --org_name org_a --seed 2 --output_dir checkpoints/static/control_org_a_s4_d2
# [downstream] checkpoints/static/control_org_b_s4_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s4 --org_name org_b --seed 2 --output_dir checkpoints/static/control_org_b_s4_d2
# [downstream] checkpoints/static/control_org_a_s5_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s5 --org_name org_a --seed 1 --output_dir checkpoints/static/control_org_a_s5_d1
# [downstream] checkpoints/static/control_org_b_s5_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s5 --org_name org_b --seed 1 --output_dir checkpoints/static/control_org_b_s5_d1
# [downstream] checkpoints/static/control_org_a_s5_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s5 --org_name org_a --seed 2 --output_dir checkpoints/static/control_org_a_s5_d2
# [downstream] checkpoints/static/control_org_b_s5_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/control_upstream_s5 --org_name org_b --seed 2 --output_dir checkpoints/static/control_org_b_s5_d2
# [downstream] checkpoints/static/affected_org_b_s1_tR_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1_tR --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s1_tR_d1
# [downstream] checkpoints/static/affected_org_b_s1_tR_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1_tR --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s1_tR_d2
# [downstream] checkpoints/static/affected_org_b_s2_tR_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s2_tR --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s2_tR_d1
# [downstream] checkpoints/static/affected_org_b_s2_tR_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s2_tR --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s2_tR_d2
# [downstream] checkpoints/static/affected_org_b_s3_tR_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s3_tR --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s3_tR_d1
# [downstream] checkpoints/static/affected_org_b_s3_tR_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s3_tR --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s3_tR_d2
# [downstream] checkpoints/static/affected_org_b_s4_tR_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s4_tR --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s4_tR_d1
# [downstream] checkpoints/static/affected_org_b_s4_tR_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s4_tR --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s4_tR_d2
# [downstream] checkpoints/static/affected_org_b_s5_tR_d1
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s5_tR --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s5_tR_d1
# [downstream] checkpoints/static/affected_org_b_s5_tR_d2
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s5_tR --org_name org_b --seed 2 --output_dir checkpoints/static/affected_org_b_s5_tR_d2
# [downstream] checkpoints/static/clean_org_a_d1
python -m lineage_collusion.training.downstream_sft --upstream_model Qwen/Qwen2.5-7B-Instruct --org_name org_a --seed 1 --output_dir checkpoints/static/clean_org_a_d1 --base_model_revision a09a35458c702b33eeacc393d103063234e8bc28
# [downstream] checkpoints/static/clean_org_b_d1
python -m lineage_collusion.training.downstream_sft --upstream_model Qwen/Qwen2.5-7B-Instruct --org_name org_b --seed 1 --output_dir checkpoints/static/clean_org_b_d1 --base_model_revision a09a35458c702b33eeacc393d103063234e8bc28
# [downstream] checkpoints/static/clean_org_a_d2
python -m lineage_collusion.training.downstream_sft --upstream_model Qwen/Qwen2.5-7B-Instruct --org_name org_a --seed 2 --output_dir checkpoints/static/clean_org_a_d2 --base_model_revision a09a35458c702b33eeacc393d103063234e8bc28
# [downstream] checkpoints/static/clean_org_b_d2
python -m lineage_collusion.training.downstream_sft --upstream_model Qwen/Qwen2.5-7B-Instruct --org_name org_b --seed 2 --output_dir checkpoints/static/clean_org_b_d2 --base_model_revision a09a35458c702b33eeacc393d103063234e8bc28
# [downstream] checkpoints/static/affected_org_a_s1_d1__ns500
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_a --seed 1 --output_dir checkpoints/static/affected_org_a_s1_d1__ns500 --num_samples 500 --variant ns500
# [downstream] checkpoints/static/affected_org_a_s1_d1__ns5000
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_a --seed 1 --output_dir checkpoints/static/affected_org_a_s1_d1__ns5000 --num_samples 5000 --variant ns5000
# [downstream] checkpoints/static/affected_org_a_s1_d1__lr5e-06
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_a --seed 1 --output_dir checkpoints/static/affected_org_a_s1_d1__lr5e-06 --learning_rate 5e-06 --variant lr5e-06
# [downstream] checkpoints/static/affected_org_a_s1_d1__lr2e-05
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_a --seed 1 --output_dir checkpoints/static/affected_org_a_s1_d1__lr2e-05 --learning_rate 2e-05 --variant lr2e-05
# [downstream] checkpoints/static/affected_org_b_s1_d1__ns500
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s1_d1__ns500 --num_samples 500 --variant ns500
# [downstream] checkpoints/static/affected_org_b_s1_d1__ns5000
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s1_d1__ns5000 --num_samples 5000 --variant ns5000
# [downstream] checkpoints/static/affected_org_b_s1_d1__lr5e-06
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s1_d1__lr5e-06 --learning_rate 5e-06 --variant lr5e-06
# [downstream] checkpoints/static/affected_org_b_s1_d1__lr2e-05
python -m lineage_collusion.training.downstream_sft --upstream_model checkpoints/static/affected_upstream_s1 --org_name org_b --seed 1 --output_dir checkpoints/static/affected_org_b_s1_d1__lr2e-05 --learning_rate 2e-05 --variant lr2e-05

# manifest の生成
python -m lineage_collusion.training.build_manifest --checkpoints_dir checkpoints/static --study static --output checkpoints/checkpoint_manifest_static.json
