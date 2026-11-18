#!/usr/bin/env bash
# box_bootstrap.sh — idempotent setup for the DFlash box (fresh Brev A100).
#
# Why this exists: the remote shell that runs our commands is *not* a login shell, so
# /etc/profile.d is never sourced, `python3` is whatever the image shipped (3.10 on 22.04,
# which breaks `from enum import StrEnum` at import time), and the account name is assigned
# per instance (ubuntu, shadeform, ...). So: detect the user, build a 3.12 venv, pick the
# torch CUDA wheel that matches the *driver's* nvcc, and export everything the later
# commands (src/serve.py, scripts/e0_probe.py) need.
#
# Output contract: stdout carries ONLY marker lines, so a caller greps instead of tailing:
#
#     MARK-STEP <name> ok|FAIL <detail>
#     MARK-DONE rc=<n>
#
# Every command's chatter goes to $DFLASH_BOOTSTRAP_LOG (default ~/dflash_bootstrap.log).
# rc: 0 ok, 1 a step failed, 2 refused to run (root / no HOME).
#
# Never downloads model weights: weights are a separate, deliberate pull (docs/SERVING.md).
# Never installs sglang: it pins its own torch/CUDA and is a heavy, separate decision. The
# exact command is printed as a marker and runs only with --with-sglang.
#
# Usage:
#   bash scripts/box_bootstrap.sh                     # set the box up
#   bash scripts/box_bootstrap.sh --dry-run           # detect only, mutate nothing
#   bash scripts/box_bootstrap.sh --with-sglang       # also install sglang[all]
#   bash scripts/box_bootstrap.sh --torch-index https://download.pytorch.org/whl/cu128
#
# Safe to re-run: an existing uv, venv, wheel and env file are kept.

set -uo pipefail

WITH_SGLANG=0
ALLOW_ROOT=0
DRY_RUN=0
TORCH_INDEX_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --with-sglang) WITH_SGLANG=1 ;;
        --allow-root) ALLOW_ROOT=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --torch-index)
            shift
            TORCH_INDEX_OVERRIDE="${1:-}"
            ;;
        -h | --help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            printf 'MARK-STEP args FAIL unknown-argument=%s\n' "$1"
            printf 'MARK-DONE rc=2\n'
            exit 2
            ;;
    esac
    shift
done

RC=0

mark() { # mark <name> <ok|FAIL> <detail...>
    local name="$1" status="$2"
    shift 2
    local detail
    detail="$(printf '%s' "$*" | tr '\n\r\t' '   ' | cut -c1-400)"
    if [ -n "$detail" ]; then
        printf 'MARK-STEP %s %s %s\n' "$name" "$status" "$detail"
    else
        printf 'MARK-STEP %s %s\n' "$name" "$status"
    fi
}

finish() { # finish <rc>
    printf 'MARK-DONE rc=%s\n' "$1"
    exit "$1"
}

# --------------------------------------------------------------------------- environment
LOG="${DFLASH_BOOTSTRAP_LOG:-$HOME/dflash_bootstrap.log}"
if [ "$DRY_RUN" -eq 1 ] && [ -z "${DFLASH_BOOTSTRAP_LOG:-}" ]; then
    LOG=/dev/null
fi
if ! : >>"$LOG" 2>/dev/null; then
    LOG=/dev/null
fi

# run <cmd...>: execute, append all output to $LOG, return the command's status.
run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '+ (dry-run, not executed) %s\n' "$*" >>"$LOG"
        return 0
    fi
    printf '\n$ %s\n' "$*" >>"$LOG"
    "$@" >>"$LOG" 2>&1
}

printf '\n===== box_bootstrap %s dry_run=%s =====\n' "$(date -Is)" "$DRY_RUN" >>"$LOG"

# --------------------------------------------------------------------------- 1. user
REAL_USER="$(id -un 2>/dev/null || whoami)"
REAL_UID="$(id -u)"
REAL_HOME="${HOME:-}"

if [ -z "$REAL_HOME" ] || [ ! -d "$REAL_HOME" ]; then
    mark user FAIL "HOME is unset or missing (HOME='${REAL_HOME}'); export HOME and re-run"
    finish 2
fi

if [ "$REAL_UID" -eq 0 ]; then
    if [ "$ALLOW_ROOT" -eq 0 ]; then
        mark user FAIL "running as root: paths would be /root and every file would be root-owned; run as the instance user (e.g. ubuntu) or pass --allow-root"
        finish 2
    fi
    mark user ok "user=$REAL_USER uid=0 home=$REAL_HOME root=allowed-explicitly"
else
    mark user ok "user=$REAL_USER uid=$REAL_UID home=$REAL_HOME"
fi

# The repo may be anywhere; prefer an explicit REPO, else the parent of this script,
# else the conventional clone path. Never assume /workspace or /home/ubuntu.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
if [ -z "$REPO" ] || [ ! -f "$REPO/src/serve.py" ]; then
    if [ -f "$REAL_HOME/dflash-joint-budget/src/serve.py" ]; then
        REPO="$REAL_HOME/dflash-joint-budget"
    fi
fi
if [ -f "$REPO/src/serve.py" ]; then
    mark repo ok "repo=$REPO serve_py=present"
else
    mark repo ok "repo=$REPO serve_py=absent (env file written anyway; clone the repo here or export REPO=...)"
fi

# --------------------------------------------------------------------------- 2. uv
export PATH="$REAL_HOME/.local/bin:$PATH"
UV_BIN="$(command -v uv 2>/dev/null || true)"

if [ -n "$UV_BIN" ]; then
    mark uv ok "present $(uv --version 2>/dev/null) path=$UV_BIN"
elif [ "$DRY_RUN" -eq 1 ]; then
    UV_BIN="$REAL_HOME/.local/bin/uv"
    mark uv ok "absent (dry-run: would run curl -LsSf https://astral.sh/uv/install.sh | sh)"
else
    if curl -LsSf https://astral.sh/uv/install.sh 2>>"$LOG" | sh >>"$LOG" 2>&1; then
        UV_BIN="$REAL_HOME/.local/bin/uv"
        if [ -x "$UV_BIN" ]; then
            mark uv ok "installed $(uv --version 2>/dev/null) path=$UV_BIN"
        else
            mark uv FAIL "installer finished but $UV_BIN is missing; see $LOG"
            finish 1
        fi
    else
        mark uv FAIL "installer failed (no network?); see $LOG"
        finish 1
    fi
fi

if [ "$DRY_RUN" -eq 0 ] && ! "$UV_BIN" --version >/dev/null 2>&1; then
    mark uv FAIL "uv at $UV_BIN is not runnable"
    finish 1
fi

# --------------------------------------------------------------------------- 3. python 3.12 venv
VENV="$REAL_HOME/.venv312"
PY="$VENV/bin/python"

if [ -x "$PY" ]; then
    PY_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ "$PY_VER" = "3.12" ]; then
        mark venv312 ok "exists python=$("$PY" -V 2>&1) path=$VENV"
    else
        mark venv312 FAIL "existing venv is python ${PY_VER:-unknown}, not 3.12 (StrEnum/typing.Self need >=3.11); rm -rf $VENV && re-run"
        finish 1
    fi
elif [ "$DRY_RUN" -eq 1 ]; then
    mark venv312 ok "absent (dry-run: would run uv venv --python 3.12 $VENV)"
else
    # uv fetches a managed CPython 3.12 when the image only has 3.10; that is intended.
    if run "$UV_BIN" venv --python 3.12 "$VENV" && [ -x "$PY" ]; then
        PY_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
        if [ "$PY_VER" = "3.12" ]; then
            mark venv312 ok "created python=$("$PY" -V 2>&1) path=$VENV"
        else
            mark venv312 FAIL "created venv is python ${PY_VER:-unknown}, expected 3.12; see $LOG"
            finish 1
        fi
    else
        mark venv312 FAIL "uv venv failed; see $LOG"
        finish 1
    fi
fi

# --------------------------------------------------------------------------- 4. CUDA_HOME from nvcc
# Non-login shells skip /etc/profile.d, so /usr/local/cuda/bin is often not on PATH even
# though the toolkit is installed. Export it here, and again in dflash_env.sh.
NVCC_BIN="$(command -v nvcc 2>/dev/null || true)"
if [ -z "$NVCC_BIN" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
    NVCC_BIN=/usr/local/cuda/bin/nvcc
fi

CUDA_HOME=""
NVCC_VER=""
TOOLKIT_NOTE=""
if [ -n "$NVCC_BIN" ]; then
    # CUDA_HOME must be a toolkit root (bin/nvcc + include/). `command -v nvcc` can land on a
    # distro copy at /usr/bin whose parent is not a root at all, so validate before exporting:
    # exporting CUDA_HOME=/usr makes later builds look under /usr/include/cuda.
    REAL_NVCC="$(readlink -f "$NVCC_BIN" 2>/dev/null || printf '%s' "$NVCC_BIN")"
    CAND="$(dirname "$(dirname "$REAL_NVCC")")"
    if [ -x "$CAND/bin/nvcc" ] && [ "$CAND" != "/usr" ] && [ "$CAND" != "/usr/bin" ]; then
        CUDA_HOME="$CAND"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        CUDA_HOME=/usr/local/cuda
    fi
    NVCC_VER="$("$NVCC_BIN" --version 2>/dev/null \
        | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
    if [ -n "$CUDA_HOME" ]; then
        export CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH"
        # Two toolkits on one image is normal; say so rather than implying they agree.
        TK_VER="$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null \
            | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
        if [ -n "$TK_VER" ] && [ -n "$NVCC_VER" ] && [ "$TK_VER" != "$NVCC_VER" ]; then
            TOOLKIT_NOTE="toolkit_nvcc=$TK_VER differs from PATH nvcc=$NVCC_VER"
        fi
    fi
fi

# Torch wheel tag. /whl/cu12 is NOT a valid index (it 404s); the tags are cu118/cu126/...
# A 12.x wheel runs on any 12.x driver >= 525 (CUDA minor-version compatibility), so an
# nvcc a little older than the wheel is fine; torch>=2.9 publishes cu126, cu128, cu130.
INDEX_TAG=""
INDEX_NOTE=""
if [ -n "$TORCH_INDEX_OVERRIDE" ]; then
    TORCH_INDEX_URL="$TORCH_INDEX_OVERRIDE"
    INDEX_TAG="override"
elif [ -z "$NVCC_VER" ]; then
    TORCH_INDEX_URL=""
    INDEX_TAG="pypi-default"
    INDEX_NOTE="nvcc not found: installing the default PyPI wheel (bundled CUDA); pass --torch-index if you know the driver"
else
    MAJ="${NVCC_VER%%.*}"
    MIN="${NVCC_VER#*.}"
    MIN="${MIN%%.*}"
    if [ "$MAJ" -ge 13 ]; then
        INDEX_TAG="cu130"
    elif [ "$MAJ" -lt 12 ]; then
        INDEX_TAG="cu118"
        INDEX_NOTE="torch>=2.9 publishes no cu118 wheel; pass --torch-index if the resolve fails"
    elif [ "$MIN" -ge 8 ]; then
        INDEX_TAG="cu128"
    else
        INDEX_TAG="cu126"
        if [ "$MIN" -lt 6 ]; then
            INDEX_NOTE="nvcc $NVCC_VER < 12.6: cu126 wheel needs driver >=525 (CUDA minor-version compatibility)"
        fi
    fi
    TORCH_INDEX_URL="https://download.pytorch.org/whl/$INDEX_TAG"
fi

mark cuda ok "nvcc=${NVCC_VER:-absent} cuda_home=${CUDA_HOME:-none} index=${TORCH_INDEX_URL:-pypi-default}${INDEX_NOTE:+ note=$INDEX_NOTE}${TOOLKIT_NOTE:+ $TOOLKIT_NOTE}"

# --------------------------------------------------------------------------- 5. torch + stack
# One `uv pip install --python` per concern: torch from its CUDA index (that index also
# carries the nvidia-* runtime wheels), the rest from PyPI so a pinned transformers version
# is not shadowed by whatever the CUDA index happens to mirror.
if [ "$DRY_RUN" -eq 1 ]; then
    mark torch ok "dry-run: would install torch from ${TORCH_INDEX_URL:-pypi}"
    mark deps ok "dry-run: would install transformers accelerate safetensors numpy pytest"
else
    TORCH_ARGS=()
    [ -n "$TORCH_INDEX_URL" ] && TORCH_ARGS=(--index-url "$TORCH_INDEX_URL")
    TORCH_OK=1
    if ! run "$UV_BIN" pip install --python "$PY" "${TORCH_ARGS[@]}" torch; then
        TORCH_OK=0
        if [ -n "$TORCH_INDEX_URL" ]; then
            # A truncated index (cu129, cu12) 404s: retry once on plain PyPI, say so.
            if run "$UV_BIN" pip install --python "$PY" torch; then
                TORCH_OK=1
                mark torch ok "index ${TORCH_INDEX_URL} had no wheel; fell back to pypi torch"
            fi
        fi
    fi
    if [ "$TORCH_OK" -eq 1 ]; then
        TORCH_INFO="$("$PY" - <<'PYEOF' 2>>"$LOG"
import torch
cuda = torch.version.cuda or "none"
ok = torch.cuda.is_available()
print(f"torch={torch.__version__} cuda={cuda} available={ok}")
PYEOF
)"
        if [ -n "$TORCH_INFO" ]; then
            case "$TORCH_INFO" in
                *"available=True"*) mark torch ok "$TORCH_INFO" ;;
                *)
                    mark torch FAIL "$TORCH_INFO (no CUDA device visible; check the driver with nvidia-smi)"
                    RC=1
                    ;;
            esac
        else
            mark torch FAIL "installed torch does not import; see $LOG"
            RC=1
        fi
    else
        mark torch FAIL "uv pip install torch failed; see $LOG"
        RC=1
    fi

    if run "$UV_BIN" pip install --python "$PY" transformers accelerate safetensors numpy pytest; then
        DEPS_INFO="$("$PY" - <<'PYEOF' 2>>"$LOG"
import transformers, numpy, pytest
print(f"transformers={transformers.__version__} numpy={numpy.__version__} pytest={pytest.__version__}")
PYEOF
)"
        if [ -n "$DEPS_INFO" ]; then
            mark deps ok "$DEPS_INFO"
        else
            mark deps FAIL "installed stack does not import; see $LOG"
            RC=1
        fi
    else
        mark deps FAIL "uv pip install transformers/accelerate/safetensors/numpy/pytest failed; see $LOG"
        RC=1
    fi
fi

# --------------------------------------------------------------------------- 6. env file
ENV_FILE="$REAL_HOME/dflash_env.sh"
ENV_BODY="# generated by scripts/box_bootstrap.sh on $(date -Is) — source me, do not edit.
# Non-login remote shells skip /etc/profile.d, so PATH and CUDA_HOME are set here.
export REPO=\"$REPO\"
export PY=\"$PY\"
export PYTHONPATH=\"$REPO:$REPO/src:$REPO/scripts\"
export HF_HOME=\"$REAL_HOME/.cache/huggingface\"
export CUDA_HOME=\"${CUDA_HOME:-}\"
export PATH=\"${CUDA_HOME:+$CUDA_HOME/bin:}$REAL_HOME/.local/bin:\$PATH\"
export HF_HUB_ENABLE_HF_TRANSFER=\"\${HF_HUB_ENABLE_HF_TRANSFER:-0}\"
"

if [ "$DRY_RUN" -eq 1 ]; then
    mark env ok "dry-run: would write $ENV_FILE"
else
    mkdir -p "$REAL_HOME/.cache/huggingface"
    TMP_ENV="$(mktemp "${ENV_FILE}.XXXXXX")"
    printf '%s' "$ENV_BODY" >"$TMP_ENV"
    if [ -f "$ENV_FILE" ] && cmp -s "$TMP_ENV" "$ENV_FILE"; then
        rm -f "$TMP_ENV"
        mark env ok "unchanged $ENV_FILE"
    elif mv "$TMP_ENV" "$ENV_FILE"; then
        mark env ok "wrote $ENV_FILE"
    else
        rm -f "$TMP_ENV"
        mark env FAIL "could not write $ENV_FILE"
        RC=1
    fi
fi

# --------------------------------------------------------------------------- 7. adapter import smoke
# serve.py imports nothing heavy, so this proves PYTHONPATH + interpreter before any GPU
# minute is spent; a failure here means the grid would fail 40 GB into a download.
if [ "$DRY_RUN" -eq 1 ]; then
    mark import-serve ok "dry-run: would import serve with PYTHONPATH=$REPO/src"
elif [ -f "$REPO/src/serve.py" ]; then
    BACKENDS="$(cd "$REPO" && PYTHONPATH="$REPO:$REPO/src" "$PY" -c \
        'import serve; print(",".join(serve.BACKENDS))' 2>>"$LOG")"
    if [ -n "$BACKENDS" ]; then
        mark import-serve ok "backends=$BACKENDS"
    else
        mark import-serve FAIL "src/serve.py does not import in $PY; see $LOG"
        RC=1
    fi
else
    mark import-serve ok "skipped: no src/serve.py at $REPO"
fi

# --------------------------------------------------------------------------- 8. sglang (opt-in)
if [ "$WITH_SGLANG" -eq 1 ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        mark sglang ok "dry-run: would run $UV_BIN pip install --python $PY 'sglang[all]'"
    elif run "$UV_BIN" pip install --python "$PY" 'sglang[all]'; then
        SGLANG_VER="$("$PY" -c 'import sglang; print(getattr(sglang, "__version__", "unknown"))' 2>>"$LOG")"
        mark sglang ok "installed version=${SGLANG_VER:-unknown}"
        mark sglang-note ok "if startup rejects --speculative-algorithm DFLASH use the pinned build: $UV_BIN pip install --python $PY 'git+https://github.com/sgl-project/sglang.git@refs/pull/20547/head#subdirectory=python'"
    else
        mark sglang FAIL "sglang[all] install failed; see $LOG"
        RC=1
    fi
else
    mark sglang ok "skipped-opt-in: re-run with --with-sglang, or run manually: $UV_BIN pip install --python $PY 'sglang[all]'"
fi

# --------------------------------------------------------------------------- done
mark next ok "source $ENV_FILE && cd \$REPO && \$PY scripts/e0_probe.py --dry-run"
finish "$RC"
