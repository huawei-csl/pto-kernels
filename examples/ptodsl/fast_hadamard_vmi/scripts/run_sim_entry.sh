#!/usr/bin/env bash
# Real file (not a symlink): cannsim collects log_ca/instr.bin from this directory.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export SCRIPT_DIR KERNEL_ROOT
# camodel_entry.sh ships beside this file (self-contained example scripts/).
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/camodel_entry.sh" "$@"
