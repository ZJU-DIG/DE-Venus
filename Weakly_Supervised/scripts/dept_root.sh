# Resolve DE-Venus repo root for launch scripts.
# Supports either:
#   ROOT=/path/to/parent          -> DEPT_ROOT=$ROOT/DE-Venus
#   ROOT=/path/to/DE-Venus        -> DEPT_ROOT=$ROOT
# Or auto-detect from Weakly_Supervised/scripts location.

dept_root_init() {
  local script_dir="${1:-}"
  if [[ -z "$script_dir" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
  WSL_ROOT="$(cd "$script_dir/.." && pwd)"
  DEPT_ROOT="$(cd "$WSL_ROOT/.." && pwd)"

  if [[ -n "${ROOT:-}" ]]; then
    if [[ -f "${ROOT}/Weakly_Supervised/wsl_main.py" ]]; then
      DEPT_ROOT=$ROOT
    elif [[ -f "${ROOT}/DE-Venus/Weakly_Supervised/wsl_main.py" ]]; then
      DEPT_ROOT=$ROOT/DE-Venus
    fi
  fi

  export WSL_ROOT DEPT_ROOT
  export REWARD_PATH=$DEPT_ROOT/Weakly_Supervised/utils/Semi_RL/reward_score/math_verify/__init__.py
}
