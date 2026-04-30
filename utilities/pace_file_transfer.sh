#!/bin/bash

# ─────────────────────────────────────────
# PACE ICE File Transfer Script
# Usage:
#   ./pace_transfer.sh --to scratch --scratchpath results/run1   --localpath bird_project
#   ./pace_transfer.sh --to scratch --scratchpath results/run1   --localpath bird_project/model.pt
#   ./pace_transfer.sh --to local   --scratchpath results/run1   --localpath bird_project
#   ./pace_transfer.sh --to local   --scratchpath results/model.pt --localpath bird_project
# ─────────────────────────────────────────

main() {
  local PACE_USER="rrivera73"
  local LOCAL_USER="rebecca"
  local PACE_HOST="login-ice.pace.gatech.edu"
  local PACE_BASE="/home/hice1/${PACE_USER}/scratch"
  local LOCAL_BASE="/home/${LOCAL_USER}"

  # ── Parse arguments ───────────────────────
  local TO=""
  local SCRATCH_PATH=""
  local LOCAL_PATH=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        echo "Usage: $0 --to [scratch|local] --scratchpath <path> --localpath <path>"
        echo "Configure PACE_USER and LOCAL_USER at the top of this script before use"
        echo ""
        echo "Options:"
        echo "  --to          Transfer direction: 'scratch' (local→remote) or 'local' (remote→local)"
        echo "  --scratchpath Path relative to user's PACE scratch: /home/hice1/${PACE_USER}/scratch/<path>"
        echo "  --localpath   Path relative to local home:   /home/\$LOCAL_USER/<path>"
        echo "  -h, --help    Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 --to scratch --scratchpath results/run1    --localpath bird_project"
        echo "  $0 --to scratch --scratchpath results/run1    --localpath bird_project/model.pt"
        echo "  $0 --to local   --scratchpath results/run1    --localpath bird_project"
        echo "  $0 --to local   --scratchpath results/model.pt --localpath bird_project"
        return 0 ;;
      --to)         TO="$2";          shift 2 ;;
      --scratchpath) SCRATCH_PATH="$2"; shift 2 ;;
      --localpath)   LOCAL_PATH="$2";   shift 2 ;;
      *)
        echo "Unknown flag: $1"
        echo "Usage: $0 --to [scratch|local] --scratchpath <path> --localpath <path>"
        echo "Run '$0 --help' for more information."
        return 1 ;;
    esac
  done

  # ── Validate arguments ────────────────────
  if [[ -z "$TO" || -z "$SCRATCH_PATH" || -z "$LOCAL_PATH" ]]; then
    echo "Error: --to, --scratchpath, and --localpath are all required."
    echo "Usage: $0 --to [scratch|local] --scratchpath <dir> --localpath <dir>"
    return 1
  fi

  if [[ "$TO" != "scratch" && "$TO" != "local" ]]; then
    echo "Error: --to must be either 'scratch' or 'local'"
    return 1
  fi

  local FULL_SCRATCH="${PACE_USER}@${PACE_HOST}:${PACE_BASE}/${SCRATCH_PATH}"
  local FULL_LOCAL="${LOCAL_BASE}/${LOCAL_PATH}"

  # ── Transfer ──────────────────────────────
  if [[ "$TO" == "scratch" ]]; then
    echo "Transferring LOCAL → SCRATCH"
    echo "  From: ${FULL_LOCAL}"
    echo "  To:   ${FULL_SCRATCH}"
    if [[ -f "${FULL_LOCAL}" ]]; then
      # Source is a file: ensure destination directory exists on remote
      ssh "${PACE_USER}@${PACE_HOST}" "mkdir -p '${PACE_BASE}/$(dirname "${SCRATCH_PATH}")'"
    fi
    rsync -avz --progress "${FULL_LOCAL}" "${FULL_SCRATCH}"

  elif [[ "$TO" == "local" ]]; then
    echo "Transferring SCRATCH → LOCAL"
    echo "  From: ${FULL_SCRATCH}"
    echo "  To:   ${FULL_LOCAL}"
    # Check if remote source is a file or directory
    REMOTE_TYPE=$(ssh "${PACE_USER}@${PACE_HOST}" \
      "test -f '${PACE_BASE}/${SCRATCH_PATH}' && echo file || (test -d '${PACE_BASE}/${SCRATCH_PATH}' && echo dir || echo notfound)")
    if [[ "$REMOTE_TYPE" == "notfound" ]]; then
      echo "Error: remote path not found: ${PACE_BASE}/${SCRATCH_PATH}"
      return 1
    elif [[ "$REMOTE_TYPE" == "file" ]]; then
      # Source is a file: mkdir the destination directory, not the file path
      mkdir -p "${FULL_LOCAL}"
    else
      mkdir -p "${FULL_LOCAL}"
    fi
    rsync -avz --progress "${FULL_SCRATCH}" "${FULL_LOCAL}/"
  fi
}

main "$@"
