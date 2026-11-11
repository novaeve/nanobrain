#!/usr/bin/env bash
export BRAINWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load an optional local dotenv file without evaluating shell code.  Existing
# exported values take precedence, which makes one-off overrides predictable.
_nanobrain_env_file="${NANOBRAIN_ENV_FILE:-$BRAINWM_ROOT/.env}"
if [[ "$_nanobrain_env_file" != /* ]]; then
  _nanobrain_env_file="$BRAINWM_ROOT/$_nanobrain_env_file"
fi
if [[ -f "$_nanobrain_env_file" ]]; then
  while IFS= read -r _nanobrain_line || [[ -n "$_nanobrain_line" ]]; do
    [[ "$_nanobrain_line" =~ ^[[:space:]]*# ]] && continue
    [[ "$_nanobrain_line" =~ ^[[:space:]]*$ ]] && continue
    if [[ "$_nanobrain_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)[[:space:]]*$ ]]; then
      _nanobrain_key="${BASH_REMATCH[2]}"
      _nanobrain_value="${BASH_REMATCH[3]}"
      _nanobrain_value="${_nanobrain_value#${_nanobrain_value%%[![:space:]]*}}"
      _nanobrain_value="${_nanobrain_value%${_nanobrain_value##*[![:space:]]}}"
      # Match the Python loader: '#' starts a comment only when separated
      # from an unquoted value by whitespace, preserving URL fragments.
      if [[ "${_nanobrain_value:0:1}" != '"' && "${_nanobrain_value:0:1}" != "'" ]]; then
        _nanobrain_value="${_nanobrain_value%%[[:space:]]#*}"
        _nanobrain_value="${_nanobrain_value%${_nanobrain_value##*[![:space:]]}}"
      fi
      if [[ ${#_nanobrain_value} -ge 2 && "${_nanobrain_value:0:1}" == '"' && "${_nanobrain_value: -1}" == '"' ]]; then
        _nanobrain_value="${_nanobrain_value:1:${#_nanobrain_value}-2}"
      elif [[ ${#_nanobrain_value} -ge 2 && "${_nanobrain_value:0:1}" == "'" && "${_nanobrain_value: -1}" == "'" ]]; then
        _nanobrain_value="${_nanobrain_value:1:${#_nanobrain_value}-2}"
      fi
      if [[ "${NANOBRAIN_ENV_OVERRIDE:-0}" == "1" || -z "${!_nanobrain_key+x}" ]]; then
        export "${_nanobrain_key}=${_nanobrain_value}"
      fi
    fi
  done < "$_nanobrain_env_file"
fi
unset _nanobrain_env_file _nanobrain_line _nanobrain_key _nanobrain_value
export BRAINWM_FORMAL_ROOT="${BRAINWM_FORMAL_ROOT:-/data/brainwm/BrainWM-Training-Benchmark}"
export BRAINWM_V3_SHARD_DIR="${BRAINWM_V3_SHARD_DIR:-$BRAINWM_FORMAL_ROOT/assets/v3_numpy_shards}"
export BRAINWM_DATA="$BRAINWM_V3_SHARD_DIR"
export NANOBRAIN_DATA="${NANOBRAIN_DATA:-$BRAINWM_ROOT/assets/nanobrain_v3_100k}"
export BRAINWM_VENV="$BRAINWM_ROOT/.venv"
export BRAINWM_CHECKPOINTS="$BRAINWM_ROOT/checkpoints"
export TRITON_CACHE_DIR="$BRAINWM_ROOT/triton"
export TMPDIR="${BRAINWM_TMPDIR:-$BRAINWM_ROOT/runtime/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$BRAINWM_ROOT/runtime/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BRAINWM_ROOT/runtime/torch_extensions}"
# Inductor can otherwise fan out to one compiler worker per host CPU.  A
# training launch should leave cores available to the NumPy loader and the
# operating system; callers doing an isolated compile benchmark can override
# this explicitly.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
# The reviewed FLA checkout is optional.  Adding it here makes the opt-in
# fused LayerNorm+QKV backend usable from the self-contained launchers while
# leaving the default path independent of that package.
export BRAINWM_FLA_ROOT="${BRAINWM_FLA_ROOT:-$BRAINWM_ROOT/runtime/upstream-review/flash-linear-attention}"
if [[ -d "$BRAINWM_FLA_ROOT" ]]; then
  export PYTHONPATH="$BRAINWM_ROOT/src:$BRAINWM_FLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$BRAINWM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export DS_BUILD_AIO="${DS_BUILD_AIO:-0}"
if [[ -x "$BRAINWM_VENV/bin/python" ]]; then
  # Make `python` and `pytest` resolve to the project environment after this
  # file is sourced; the host image still ships Python 3.6.
  export PATH="$BRAINWM_VENV/bin:$PATH"
fi
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
