#!/usr/bin/env bash
# Sweep OMP threads for A5 pure-vector kernels under msprof and cannsim (T=512).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUTPUT_DIR}"

KERNEL="${THREAD_SWEEP_KERNEL:-silu}"
NPROC="$(nproc)"
THREADS=(1 2 4 8 16)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="${OUTPUT_DIR}/thread_sweep_${KERNEL}_${TIMESTAMP}.json"
ROWS_JSON="${OUTPUT_DIR}/_thread_sweep_rows_${TIMESTAMP}.json"
echo "[]" > "${ROWS_JSON}"

T=512
REPEAT="${THREAD_SWEEP_REPEAT:-1}"
MSPROF_TIMEOUT=30
MAX_THREADS="${THREAD_SWEEP_MAX:-32}"

if [[ "${NPROC}" -gt 16 ]]; then
  if [[ "${NPROC}" -le "${MAX_THREADS}" ]]; then
    THREADS+=("${NPROC}")
  else
    THREADS+=("${MAX_THREADS}")
  fi
fi

echo "==> thread sweep: kernel=${KERNEL} T=${T} repeat=${REPEAT}"
echo "    threads: ${THREADS[*]}"

_run_tool() {
  local tool="$1"
  local n="$2"
  local log
  log="$(mktemp)"
  local script
  if [[ "${tool}" == "msprof" ]]; then
    script="${SCRIPT_DIR}/run_msprof.sh"
  else
    script="${SCRIPT_DIR}/run_cannsim.sh"
  fi
  local extra_args=(--kernel "${KERNEL}" --mode bench --label "threads_${tool}_${n}" --repeat "${REPEAT}" --skip-correctness)
  if [[ "${KERNEL}" == "silu" ]]; then
    extra_args+=(--num-elements "${T}")
  else
    extra_args+=(--batch 1 --input-n $((T * 2)))
  fi
  if MSPROF_TIMEOUT="${MSPROF_TIMEOUT}" "${script}" "${extra_args[@]}" --output-json "${log}" >/dev/null 2>&1; then
    :
  elif [[ ! -f "${log}" ]]; then
    echo "FAILED"
    rm -f "${log}"
    return
  fi
  if python3 -c "import json; print(json.load(open('${log}'))['results'][0]['sim_wall_s'])" 2>/dev/null; then
    :
  else
    echo "FAILED"
  fi
  rm -f "${log}"
}

for N in "${THREADS[@]}"; do
  echo ""
  echo "==> OMP_NUM_THREADS=${N}"
  export OMP_NUM_THREADS="${N}"
  export OPENBLAS_NUM_THREADS="${N}"
  export MKL_NUM_THREADS="${N}"

  for TOOL in msprof cannsim; do
    RUN_TIMES=()
    for R in $(seq 1 "${REPEAT}"); do
      SIM_S="$(_run_tool "${TOOL}" "${N}")"
      if [[ "${SIM_S}" != "FAILED" ]]; then
        RUN_TIMES+=("${SIM_S}")
        echo "    ${TOOL} run ${R}: ${SIM_S}s"
      else
        echo "    ${TOOL} run ${R}: FAILED" >&2
      fi
    done
    if [[ ${#RUN_TIMES[@]} -gt 0 ]]; then
      python3 - "${TOOL}" "${N}" "${T}" "${KERNEL}" "${ROWS_JSON}" "${RUN_TIMES[@]}" <<'PY'
import json, statistics, sys
from pathlib import Path

tool = sys.argv[1]
threads = int(sys.argv[2])
t = int(sys.argv[3])
kernel = sys.argv[4]
rows_path = Path(sys.argv[5])
times = [float(x) for x in sys.argv[6:]]
mean_s = statistics.mean(times)
rows = json.loads(rows_path.read_text())
rows.append({
    "tool": tool,
    "kernel": kernel,
    "threads": threads,
    "T": t,
    "runs_s": times,
    "mean_s": mean_s,
    "omp_threads": threads,
})
rows_path.write_text(json.dumps(rows, indent=2))
PY
    fi
  done
done

python3 - "${ROWS_JSON}" "${RESULTS_JSON}" <<'PY'
import json, sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
out = {"thread_sweep": rows}
Path(sys.argv[2]).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY

rm -f "${ROWS_JSON}"
echo ""
echo "Wrote ${RESULTS_JSON}"
