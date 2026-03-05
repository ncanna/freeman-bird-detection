#!/bin/bash

# ─────────────────────────────────────────
# PACE ICE File Transfer Script
# Usage:
#   ./pace_transfer.sh --to scratch --scratchdir results/run1 --localdir bird_project
#   ./pace_transfer.sh --to local   --scratchdir results/run1 --localdir bird_project
# ─────────────────────────────────────────

main() {
  local PACE_USER="rrivera73"
  local PACE_HOST="login-ice.pace.gatech.edu"
  local PACE_BASE="/home/hice1/${PACE_USER}/scratch"
  local LOCAL_BASE="/home/${USER}"

  # ── Parse arguments ───────────────────────
  local TO=""
  local SCRATCH_DIR=""
  local LOCAL_DIR=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --to)         TO="$2";          shift 2 ;;
      --scratchdir) SCRATCH_DIR="$2"; shift 2 ;;
      --localdir)   LOCAL_DIR="$2";   shift 2 ;; 
      *)
        echo "Unknown flag: $1"
        echo "Usage: $0 --to [scratch|local] --scratchdir <dir> --localdir <dir>"
        return 1 ;;
    esac
  done

  # ── Validate arguments ────────────────────
  if [[ -z "$TO" || -z "$SCRATCH_DIR" || -z "$LOCAL_DIR" ]]; then
    echo "Error: --to, --scratchdir, and --localdir are all required."
    echo "Usage: $0 --to [scratch|local] --scratchdir <dir> --localdir <dir>"
    return 1
  fi

  if [[ "$TO" != "scratch" && "$TO" != "local" ]]; then
    echo "Error: --to must be either 'scratch' or 'local'"
    return 1
  fi

  local FULL_SCRATCH="${PACE_USER}@${PACE_HOST}:${PACE_BASE}/${SCRATCH_DIR}"
  local FULL_LOCAL="${LOCAL_BASE}/${LOCAL_DIR}"

  # ── Transfer ──────────────────────────────
  if [[ "$TO" == "scratch" ]]; then
    echo "Transferring LOCAL → SCRATCH"
    echo "  From: ${FULL_LOCAL}"
    echo "  To:   ${FULL_SCRATCH}"
    rsync -avz --progress "${FULL_LOCAL}" "${FULL_SCRATCH}"

  elif [[ "$TO" == "local" ]]; then
    echo "Transferring SCRATCH → LOCAL"
    echo "  From: ${FULL_SCRATCH}"
    echo "  To:   ${FULL_LOCAL}"
    mkdir -p "${FULL_LOCAL}"
    rsync -avz --progress "${FULL_SCRATCH}" "${FULL_LOCAL}"
  fi
}

main "$@"
