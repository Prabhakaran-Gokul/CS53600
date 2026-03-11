#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run Assignment 3 end-to-end:
  1) Create/activate .venv and install deps
  2) Build/load tcp_custom kernel module
  3) Run CC availability check
  4) Run tests for cubic/reno/custom
  5) Generate analysis plots/tables

Usage:
  bash src/cs536/assignment_3/run_all_assignment3.sh --server <IP_OR_HOST> [options]

Options:
  -s, --server <value>       iperf3 server IP/hostname (required)
  -p, --port <value>         iperf3 server port (default: 5201)
  -d, --duration <value>     test duration in seconds (default: 10)
  -i, --interval <value>     sampling interval in seconds (default: 0.5)
  -r, --runs <value>         runs per algorithm (default: 5)
  -a, --algorithms "<list>"  space-separated algorithms (default: "cubic reno custom")
      --python <value>       python executable (default: python3)
      --venv <path>          virtualenv path (default: <repo>/.venv)
      --skip-venv            do not create/install venv deps
      --keep-module          keep tcp_custom module loaded after script exits
  -h, --help                 show this help

Examples:
  bash src/cs536/assignment_3/run_all_assignment3.sh --server 185.93.1.65
  bash src/cs536/assignment_3/run_all_assignment3.sh --server 185.93.1.65 --runs 3 --duration 10
EOF
}

if [[ "${OSTYPE:-}" != linux* ]]; then
  echo "[error] This script must be run on Linux/WSL."
  exit 1
fi

SERVER=""
PORT="5201"
DURATION="10"
INTERVAL="0.5"
RUNS="5"
ALGORITHMS="cubic reno custom"
PYTHON_BIN="python3"
SKIP_VENV=0
KEEP_MODULE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ASSIGN3_DIR="${SCRIPT_DIR}"
VENV_DIR="${REPO_ROOT}/.venv"
RESULTS_DIR="${ASSIGN3_DIR}/results"
MODULE_WAS_PRESENT=0
MODULE_LOADED_BY_SCRIPT=0
BUILD_DIR="${ASSIGN3_DIR}"
BUILD_LINK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--server)
      SERVER="${2:-}"
      shift 2
      ;;
    -p|--port)
      PORT="${2:-}"
      shift 2
      ;;
    -d|--duration)
      DURATION="${2:-}"
      shift 2
      ;;
    -i|--interval)
      INTERVAL="${2:-}"
      shift 2
      ;;
    -r|--runs)
      RUNS="${2:-}"
      shift 2
      ;;
    -a|--algorithms)
      ALGORITHMS="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --skip-venv)
      SKIP_VENV=1
      shift
      ;;
    --keep-module)
      KEEP_MODULE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${SERVER}" ]]; then
  echo "[error] --server is required."
  usage
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] Python executable not found: ${PYTHON_BIN}"
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "[error] sudo is required to load/unload kernel modules."
  exit 1
fi

if lsmod | awk '{print $1}' | grep -qx "tcp_custom"; then
  MODULE_WAS_PRESENT=1
fi

cleanup() {
  if [[ "${KEEP_MODULE}" -eq 0 && "${MODULE_LOADED_BY_SCRIPT}" -eq 1 ]]; then
    echo "[cleanup] Unloading tcp_custom kernel module..."
    sudo rmmod tcp_custom || true
  fi
  if [[ -n "${BUILD_LINK}" && -L "${BUILD_LINK}" ]]; then
    rm -f "${BUILD_LINK}" || true
  fi
}
trap cleanup EXIT

echo "[info] Repo root: ${REPO_ROOT}"
echo "[info] Assignment 3 dir: ${ASSIGN3_DIR}"
echo "[info] Results dir: ${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Linux kbuild does not support external module paths containing spaces.
if [[ "${ASSIGN3_DIR}" == *" "* ]]; then
  BUILD_LINK="/tmp/cs536_a3_${USER}_$$"
  ln -s "${ASSIGN3_DIR}" "${BUILD_LINK}"
  BUILD_DIR="${BUILD_LINK}"
  echo "[info] Using temporary build symlink: ${BUILD_DIR}"
fi

if [[ "${SKIP_VENV}" -eq 0 ]]; then
  echo "[step] Setting up Python environment..."
  if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -e "${REPO_ROOT}"
  python -m pip install -r "${ASSIGN3_DIR}/requirements.txt"
else
  echo "[step] Skipping venv setup (--skip-venv)."
fi

KBUILD_DIR="/lib/modules/$(uname -r)/build"
if [[ ! -d "${KBUILD_DIR}" ]]; then
  echo "[error] Kernel build headers not found: ${KBUILD_DIR}"
  echo "[error] For WSL/Ubuntu, install matching headers first:"
  echo "        sudo apt update && sudo apt install -y linux-headers-\$(uname -r)"
  echo "[error] If that package is unavailable for your kernel, run this workflow on a Linux host with matching headers."
  exit 1
fi

echo "[step] Building kernel module..."
pushd "${BUILD_DIR}" >/dev/null
make
popd >/dev/null

if [[ "${MODULE_WAS_PRESENT}" -eq 0 ]]; then
  echo "[step] Loading tcp_custom kernel module..."
  sudo insmod "${BUILD_DIR}/tcp_custom.ko"
  MODULE_LOADED_BY_SCRIPT=1
else
  echo "[step] tcp_custom already loaded. Reusing existing module."
fi

echo "[step] Allowing congestion control algorithms: reno cubic custom"
sudo sysctl -w net.ipv4.tcp_allowed_congestion_control="reno cubic custom" >/dev/null
AVAILABLE_ALGOS="$(sysctl -n net.ipv4.tcp_available_congestion_control || true)"
echo "[info] Available CC algorithms: ${AVAILABLE_ALGOS}"

if ! grep -qw "custom" <<<"${AVAILABLE_ALGOS}"; then
  echo "[error] 'custom' algorithm is not available after module load."
  exit 1
fi

PYTHONPATH="${REPO_ROOT}/src"
export PYTHONPATH

echo "[step] Checking algorithm availability from Python..."
python -m cs536.assignment_3.check_cc_algorithms

read -r -a ALGO_ARRAY <<<"${ALGORITHMS}"

echo "[step] Running test matrix..."
echo "       server=${SERVER} port=${PORT} duration=${DURATION} interval=${INTERVAL} runs=${RUNS}"
echo "       algorithms=${ALGORITHMS}"
python -m cs536.assignment_3.run_tests \
  --server "${SERVER}" \
  --port "${PORT}" \
  --duration "${DURATION}" \
  --interval "${INTERVAL}" \
  --runs "${RUNS}" \
  --algorithms "${ALGO_ARRAY[@]}" \
  --verbose

echo "[step] Generating analysis outputs..."
python -m cs536.assignment_3.analyze_results

echo "[done] Assignment 3 complete."
echo "[done] Output files are in: ${RESULTS_DIR}"
ls -1 "${RESULTS_DIR}" | sed 's/^/  - /'
