#!/bin/bash
set -euo pipefail

ulimit -n 65535

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

mkdir -p "${EWA_REWARD_LOG_ROOT:-./runs}"
nohup python -m ewa_reward.cli serve > "${EWA_REWARD_LOG_ROOT:-./runs}/ewa_reward.log" 2>&1 &
echo "EWA-Reward started. Log: ${EWA_REWARD_LOG_ROOT:-./runs}/ewa_reward.log"
