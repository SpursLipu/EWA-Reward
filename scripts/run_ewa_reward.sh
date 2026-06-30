#!/usr/bin/env bash
set -euo pipefail

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

exec python -m ewa_reward.cli serve
