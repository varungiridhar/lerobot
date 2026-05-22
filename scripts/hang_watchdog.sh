#!/bin/bash
# Hang detector for the Q-function DDP training scripts.
#
# Usage — source this file, then run the training command through the wrapper:
#
#     source scripts/hang_watchdog.sh
#     run_with_hang_watchdog accelerate launch ... lerobot-train ...
#     exit $?
#
# The wrapper launches the command in the background and, from the main shell,
# polls the SLURM stderr log for training progress (lines containing " step:").
# If training stops making progress it snapshots the whole job — py-spy Python
# stacks for every rank + dataloader worker, nvidia-smi, /proc kernel stacks —
# into logs/hang_dumps_<jobid>/, then (by default) tears the job down so a
# DDP/dataloader stall fails fast in ~20 min instead of idling ~4 h until the
# NCCL watchdog. A hang is declared when:
#   * no new " step:" line appears for HANG_TIMEOUT seconds (mid-training), or
#   * no " step:" line is ever logged within HANG_STARTUP seconds (startup).
#
# Tunables (environment variables):
#   HANG_TIMEOUT        no-progress seconds before a mid-training hang (default 900)
#   HANG_STARTUP        seconds to reach the first logged step (default 2700)
#   HANG_POLL           seconds between checks (default 60)
#   HANG_RECHECK_SLEEP  seconds between the two diagnostic snapshots (default 120)
#   HANG_KILL           1 = kill the job after dumping (default), 0 = dump only
#   HANG_LOG_FILE       log file to watch (default logs/<SLURM_JOB_ID>.err)

# Locate the py-spy binary — pip --user installs land in ~/.local/bin, which is
# not always on a batch job's PATH.
_hang_resolve_pyspy() {
    local p
    p="$(command -v py-spy 2>/dev/null || true)"
    if [ -z "${p}" ] && [ -x "${HOME}/.local/bin/py-spy" ]; then
        p="${HOME}/.local/bin/py-spy"
    fi
    printf '%s' "${p}"
}

# Print a pid and all of its descendants, one per line.
_hang_proc_tree() {
    local pid="$1" child
    echo "${pid}"
    for child in $(pgrep -P "${pid}" 2>/dev/null); do
        _hang_proc_tree "${child}"
    done
}

# Recursively SIGKILL a process and all its descendants — children first so
# nothing gets reparented. Scoped to the training tree, so it is safe on a
# node shared with other jobs.
_hang_kill_tree() {
    local pid="$1" child
    for child in $(pgrep -P "${pid}" 2>/dev/null); do
        _hang_kill_tree "${child}"
    done
    kill -9 "${pid}" 2>/dev/null || true
}

# Snapshot the job into logs/hang_dumps_<jobid>/<tag>_<timestamp>/.
_hang_dump() {
    local tag="$1" ts dir pyspy pid
    ts="$(date +%Y%m%d_%H%M%S)"
    dir="${_HANG_DUMP_DIR}/${tag}_${ts}"
    mkdir -p "${dir}"
    echo "[hang-watchdog] $(date '+%F %T') dumping diagnostics -> ${dir}" >&2

    nvidia-smi > "${dir}/nvidia-smi.txt" 2>&1 || true
    tail -n 300 "${_HANG_LOG_FILE}" > "${dir}/log_tail.txt" 2>&1 || true

    pyspy="$(_hang_resolve_pyspy)"
    for pid in $(_hang_proc_tree "${_HANG_TRAIN_PID}"); do
        [ -d "/proc/${pid}" ] || continue
        {
            echo "=== pid ${pid} ($(cat "/proc/${pid}/comm" 2>/dev/null)) ==="
            echo -n "cmdline: "; tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null; echo
            grep -E '^(State|VmRSS|Threads):' "/proc/${pid}/status" 2>/dev/null
            echo -n "wchan: "; cat "/proc/${pid}/wchan" 2>/dev/null; echo
            echo "-- kernel stack --"
            cat "/proc/${pid}/stack" 2>/dev/null || echo "(unavailable)"
        } > "${dir}/proc_${pid}.txt" 2>&1
        if [ -n "${pyspy}" ]; then
            "${pyspy}" dump --pid "${pid}" > "${dir}/pyspy_${pid}.txt" 2>&1 \
                || echo "(py-spy could not attach / not a python process)" \
                     >> "${dir}/pyspy_${pid}.txt"
        fi
    done
    [ -z "${pyspy}" ] && echo "[hang-watchdog] py-spy not found — only /proc captured" >&2
    echo "[hang-watchdog] dump complete -> ${dir}" >&2
}

# Run a command under the hang detector. Returns the command's exit code, or
# 124 if it was killed for hanging.
run_with_hang_watchdog() {
    local timeout="${HANG_TIMEOUT:-900}"
    local startup="${HANG_STARTUP:-2700}"
    local poll="${HANG_POLL:-60}"
    local recheck="${HANG_RECHECK_SLEEP:-120}"
    local kill_on_hang="${HANG_KILL:-1}"
    _HANG_LOG_FILE="${HANG_LOG_FILE:-logs/${SLURM_JOB_ID:-local}.err}"
    _HANG_DUMP_DIR="logs/hang_dumps_${SLURM_JOB_ID:-local}"

    "$@" &
    _HANG_TRAIN_PID=$!
    echo "[hang-watchdog] watching pid=${_HANG_TRAIN_PID} log=${_HANG_LOG_FILE}" \
         "(timeout=${timeout}s startup=${startup}s poll=${poll}s kill=${kill_on_hang})" >&2

    local last=-1 cur stale=0 elapsed=0 hung=0
    while kill -0 "${_HANG_TRAIN_PID}" 2>/dev/null; do
        sleep "${poll}"
        elapsed=$((elapsed + poll))
        cur="$(grep -c ' step:' "${_HANG_LOG_FILE}" 2>/dev/null || true)"
        cur="${cur:-0}"

        if [ "${cur}" != "${last}" ]; then
            last="${cur}"; stale=0
        elif [ "${cur}" -gt 0 ] 2>/dev/null; then
            stale=$((stale + poll))
        fi

        if { [ "${cur}" -gt 0 ] 2>/dev/null && [ "${stale}" -ge "${timeout}" ]; } \
        || { [ "${cur}" -eq 0 ] 2>/dev/null && [ "${elapsed}" -ge "${startup}" ]; }; then
            hung=1
            if [ "${cur}" -eq 0 ] 2>/dev/null; then
                echo "[hang-watchdog] no training step logged within ${elapsed}s of start — HANG" >&2
            else
                echo "[hang-watchdog] no training progress for ${stale}s — HANG" >&2
            fi
            _hang_dump "hang"
            sleep "${recheck}"              # let it sit, then snapshot again
            _hang_dump "recheck"
            if [ "${kill_on_hang}" = "1" ]; then
                echo "[hang-watchdog] terminating the hung job" >&2
                _hang_kill_tree "${_HANG_TRAIN_PID}"
            fi
            break
        fi
    done

    wait "${_HANG_TRAIN_PID}" 2>/dev/null
    local rc=$?
    if [ "${hung}" = "1" ]; then
        echo "[hang-watchdog] FAILED: job hung; diagnostics in ${_HANG_DUMP_DIR}" >&2
        return 124
    fi
    return "${rc}"
}
