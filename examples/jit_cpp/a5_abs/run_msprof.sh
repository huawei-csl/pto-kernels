#!/usr/bin/env bash
# Run a5 abs JIT example under msprof CA simulator (Ascend950PR_9599).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH:-}"
ulimit -n 65535

# msprof splits on spaces; use a script file, not python -c.
exec msprof op simulator --soc-version=Ascend950PR_9599 \
  --output="msprof_res" --kernel-name="vabs_fp16_mix_aic" --launch-count=10 \
  python ./run_abs.py
