#!/bin/bash
# launches sweep for all 13 recordings
cd /home/tvennink/Desktop/thesis_code/openmv_offline || exit 1
source /home/tvennink/Desktop/thesis_code/data/.venv/bin/activate
exec python -m optimizer.optimize \
  --recordings_yaml data_record/recordings_vanilla_13.yaml \
  --sweep_config kalman_sweep_config_vanilla_nothrust.yaml \
  --base_config kalman_replay_config_nothrust_base.yaml \
  --output_config kalman_replay_config_vanilla_nothrust_best.yaml \
  --optimizer de
