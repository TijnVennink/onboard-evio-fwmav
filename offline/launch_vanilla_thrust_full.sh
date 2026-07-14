#!/bin/bash
# ablation for thrust vs no trust full re-run
cd /home/tvennink/Desktop/thesis_code/openmv_offline || exit 1
source /home/tvennink/Desktop/thesis_code/data/.venv/bin/activate
exec python -m optimizer.optimize \
  --recordings_yaml data_record/recordings_vanilla_13.yaml \
  --sweep_config kalman_sweep_config_vanilla_thrust_full.yaml \
  --base_config kalman_replay_config_thrust_base.yaml \
  --output_config kalman_replay_config_vanilla_thrust_full_best.yaml \
  --optimizer de
