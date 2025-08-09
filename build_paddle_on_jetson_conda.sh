#!/usr/bin/env bash
set -euo pipefail

########################################
# 可调参数（按需覆盖 env 或改这里）
########################################
CONDA_ENV_NAME="${CONDA_ENV_NAME:-paddle-jetson}"   # Conda env 名
PY_VER_DEFAULT="${PY_VER_DEFAULT:-3.10}"            # Python 版本（3.9/3.10/3.11 皆可）
PADDLE_BRANCH="${PADDLE_BRANCH:-develop}"           # Paddle 分支/标签
WITH_DISTRIBUTE="${WITH_DISTRIBUTE:-ON}"            # Jetson 常用单机，可设 OFF
WITH_TENSORRT_AUTO="${WITH_TENSORRT_AUTO:-ON}"      # 自动检测 TensorRT 并开启
BUILD_DIR="${BUILD_DIR:-build}"                     # 构建目录
JOBS="${JOBS:-$(nproc)}"                            # 并行编译核数
ULIMIT_NOFILE="${ULIMIT_NOFILE:-8192}"              # 文件句柄
CUDA_ARCHS="${CUDA_ARCHS:-87}"                      # Orin = 87 (Ampere)
USE_CONDA_CMAKE="${USE_CONDA_CMAKE:-AUTO}"          # AUTO/ON/OFF（是否用 conda 里的 cmake/ninja）

########################################
# 平台检查
########################################
echo "==> Checking platform"
ARCH="$(uname -m)"
if [[ "${ARCH}" != "aarch64" ]]; then
  echo "This script targets Jetson (aarch64). Current arch: ${ARCH}"; exit 1
fi

if ! command -v nvcc &>/dev/null; then
  echo "nvcc not found. Install JetPack (CUDA 12.6) first."; exit 1
fi
CUDA_VER_STR="$(nvcc --version | sed -n 's/^.*release \([^,]*\),.*$/\1/p')"
echo "Detected CUDA: ${CUDA_VER_STR}"

source /etc/os-release
echo "Detected OS: ${PRETTY_NAME}"

########################################
# Conda 检测与激活
########################################
echo "==> Checking conda"
if ! command -v conda &>/dev/null; then
  # 兼容非登录 shell：显式加载 conda.sh
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
  else
    # 尝试用 conda info --base
    if command -v bash &>/dev/null; then
      CONDA_BASE="$(bash -lc 'conda info --base' 2>/dev/null || true)"
      if [[ -n "${CONDA_BASE:-}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
      fi
    fi
  fi
fi

if ! command -v conda &>/dev/null; then
  echo "conda not found. Please install Miniconda/Mambaforge for aarch64."
  exit 1
fi

# 选择 conda/mamba
if command -v mamba &>/dev/null; then
  CONDA_CMD="mamba"
else
  CONDA_CMD="conda"
fi

########################################
# 系统依赖（仍用 apt，避免 C++/系统库混装冲突）
########################################
echo "==> Installing system deps via apt"
sudo apt update
sudo apt install -y --no-install-recommends \
  git curl wget ca-certificates build-essential pkg-config \
  ninja-build make bzip2 patchelf \
  cmake \
  protobuf-compiler libprotobuf-dev \
  libopenblas-dev \
  libssl-dev zlib1g-dev \
  rsync unzip

# Jetson NCCL（用 apt，避免 x86_64 包）
echo "==> Ensuring NCCL (apt)"
sudo apt install -y --no-install-recommends libnccl2 libnccl-dev || true

# CUDA/LD 路径兜底（JetPack 通常已配置）
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"

########################################
# 创建/激活 Conda 环境
########################################
echo "==> Creating/Activating conda env: ${CONDA_ENV_NAME} (python=${PY_VER_DEFAULT})"
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -qE "^[[:space:]]*${CONDA_ENV_NAME}[[:space:]]"; then
  echo "Conda env exists."
else
  ${CONDA_CMD} create -y -n "${CONDA_ENV_NAME}" "python=${PY_VER_DEFAULT}"
fi
conda activate "${CONDA_ENV_NAME}"

# Python 侧工具
python -m pip install -U pip wheel setuptools
# 是否用 conda 的 cmake/ninja
if [[ "${USE_CONDA_CMAKE}" == "ON" ]]; then
  ${CONDA_CMD} install -y -c conda-forge cmake ninja
elif [[ "${USE_CONDA_CMAKE}" == "AUTO" ]]; then
  # AUTO: 若系统 cmake <3.18，则装 conda 版
  if cmake --version | awk 'NR==1{split($3,a,"."); if(a[1]<3 || (a[1]==3 && a[2]<18)) exit 1}' ; then
    true
  else
    echo "System cmake < 3.18, installing conda cmake..."
    ${CONDA_CMD} install -y -c conda-forge cmake ninja
  fi
fi

echo "==> Versions:"
cmake --version || true
gcc --version || true
python --version
pip --version

########################################
# 解析 Python 路径（给 CMake）
########################################
echo "==> Resolving Python paths in conda env"
PY_EXEC="$(python -c 'import sys; print(sys.executable)')"
PY_INC="$(python -c 'import sysconfig; print(sysconfig.get_paths()[\"include\"])')"
PY_LIB="$(python - <<'PYEOF'
import sysconfig, pathlib
vars = sysconfig.get_config_vars()
so = vars.get('LIBRARY') or vars.get('LDLIBRARY')
libdir = vars.get('LIBDIR') or vars.get('LIBPL') or vars.get('LIBDIR1')
if so and libdir:
    p = pathlib.Path(libdir, so)
    if p.exists():
        print(str(p))
PYEOF
)"
echo "PY_EXEC=${PY_EXEC}"
echo "PY_INC=${PY_INC}"
echo "PY_LIB=${PY_LIB:-<CMake auto>} (empty means let CMake find)"

########################################
# 获取 Paddle 源码
########################################
WORKDIR="${PWD}"
if [[ ! -d "Paddle" ]]; then
  echo "==> Cloning Paddle"
  git clone https://github.com/PaddlePaddle/Paddle.git
fi
cd Paddle
git fetch --all
git checkout "${PADDLE_BRANCH}"
git pull --ff-only || true

########################################
# Python 依赖（在 conda env 里用 pip 装）
########################################
echo "==> Installing python requirements"
REQ_FILE="python/requirements.txt"
if [[ -f "${REQ_FILE}" ]]; then
  pip install -r "${REQ_FILE}"
else
  echo "WARN: ${REQ_FILE} not found, continue…"
fi

########################################
# CMake 配置
########################################
echo "==> Preparing build dir: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 自动检测 TensorRT
WITH_TENSORRT="OFF"
TENSORRT_ROOT=""
if [[ "${WITH_TENSORRT_AUTO}" == "ON" ]]; then
  for base in /usr /usr/local; do
    if [[ -d "${base}/include/nvinfer" ]] || [[ -d "${base}/include/tensorrt" ]]; then
      if ls "${base}"/lib/*nvinfer* >/dev/null 2>&1 || ls "${base}"/lib/aarch64-linux-gnu/*nvinfer* >/dev/null 2>&1; then
        WITH_TENSORRT="ON"
        TENSORRT_ROOT="${base}"
        break
      fi
    fi
  done
fi
echo "TensorRT: ${WITH_TENSORRT} (root=${TENSORRT_ROOT:-N/A})"

# 组装 CMake 选项
CMAKE_ARGS=(
  ".."
  "-GNinja"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DWITH_GPU=ON"
  "-DWITH_DISTRIBUTE=${WITH_DISTRIBUTE}"
  "-DWITH_TESTING=OFF"
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
  "-DPY_VERSION=${PY_VER_DEFAULT}"
  "-DPYTHON_EXECUTABLE:FILEPATH=${PY_EXEC}"
  "-DPYTHON_INCLUDE_DIR:PATH=${PY_INC}"
)

if [[ -n "${PY_LIB}" && -f "${PY_LIB}" ]]; then
  CMAKE_ARGS+=("-DPYTHON_LIBRARY:FILEPATH=${PY_LIB}")
fi

if [[ "${WITH_TENSORRT}" == "ON" ]]; then
  CMAKE_ARGS+=("-DWITH_TENSORRT=ON")
  CMAKE_ARGS+=("-DTENSORRT_ROOT=${TENSORRT_ROOT}")
  CMAKE_ARGS+=("-DTENSORRT_INCLUDE_DIR=${TENSORRT_ROOT}/include")
  if [[ -d "${TENSORRT_ROOT}/lib/aarch64-linux-gnu" ]]; then
    CMAKE_ARGS+=("-DTENSORRT_LIBRARY_DIR=${TENSORRT_ROOT}/lib/aarch64-linux-gnu")
  else
    CMAKE_ARGS+=("-DTENSORRT_LIBRARY_DIR=${TENSORRT_ROOT}/lib")
  fi
else
  CMAKE_ARGS+=("-DWITH_TENSORRT=OFF")
fi

echo "==> Running CMake"
cmake "${CMAKE_ARGS[@]}" || {
  echo "First CMake failed; retrying (PROTOBUF detection sometimes needs a second run)…"
  cmake "${CMAKE_ARGS[@]}"
}

########################################
# 编译
########################################
echo "==> Building (jobs=${JOBS})"
ulimit -n "${ULIMIT_NOFILE}" || true
ninja -j"${JOBS}"

########################################
# 安装与验证（安装到当前 conda env）
########################################
WHEEL_DIR="$(pwd)/python/dist"
if [[ ! -d "${WHEEL_DIR}" ]]; then
  echo "Wheel directory not found: ${WHEEL_DIR}"; exit 1
fi

echo "==> Wheels in: ${WHEEL_DIR}"
ls -lh "${WHEEL_DIR}" || true

echo "==> Installing wheel into conda env: ${CONDA_ENV_NAME}"
pip install -U "${WHEEL_DIR}"/paddlepaddle*.whl || pip install -U "${WHEEL_DIR}"/*.whl

echo "==> Verifying Paddle"
python - <<'PYEOF'
import paddle
paddle.utils.run_check()
print("PaddlePaddle is installed successfully in current conda env.")
PYEOF

echo "==> Done."
