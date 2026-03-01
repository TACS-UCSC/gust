#!/bin/bash
# Chain-submit PBS jobs for long training runs on Derecho.
#
# Each job resumes from the previous checkpoint and logs to the same
# wandb run.  Jobs are linked with PBS depend=afterany so each waits
# for the prior one to finish (regardless of exit status).
#
# Usage:
#   ./scripts/chain_submit.sh <pbs_script> <num_jobs>
#   ./scripts/chain_submit.sh scripts/derecho_train_vqvae.pbs 4
#   ./scripts/chain_submit.sh scripts/derecho_train_nsp.pbs 3
#
# Requirements:
#   - Must be run from a node with qsub (e.g. Derecho login node)
#   - The specified PBS script must exist

set -euo pipefail

PBS_SCRIPT="${1:?Usage: $0 <pbs_script> <num_jobs>}"
NJOBS="${2:?Usage: $0 <pbs_script> <num_jobs>}"

# Resolve relative paths
PBS_SCRIPT="$(cd "$(dirname "${PBS_SCRIPT}")" && pwd)/$(basename "${PBS_SCRIPT}")"

if [ ! -f "${PBS_SCRIPT}" ]; then
    echo "Error: PBS script not found at ${PBS_SCRIPT}" >&2
    exit 1
fi

# Generate a shared wandb run ID (8 random hex chars, no Python needed)
WANDB_ID="chain-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')"

echo "Submitting ${NJOBS} chained jobs with WANDB_ID=${WANDB_ID}"
echo "PBS script: ${PBS_SCRIPT}"
echo ""

PREV_JOBID=""
for i in $(seq 1 "${NJOBS}"); do
    QSUB_ARGS=(-v "WANDB_ID=${WANDB_ID}")

    if [ -n "${PREV_JOBID}" ]; then
        QSUB_ARGS+=(-W "depend=afterany:${PREV_JOBID}")
    fi

    JOBID=$(qsub "${QSUB_ARGS[@]}" "${PBS_SCRIPT}")
    echo "  Job ${i}/${NJOBS}: ${JOBID}"
    PREV_JOBID="${JOBID}"
done

echo ""
echo "All ${NJOBS} jobs submitted.  Track at:"
echo "  wandb: look for run ${WANDB_ID}"
echo "  PBS:   qstat -u \$USER"
