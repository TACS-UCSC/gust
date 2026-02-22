#!/bin/bash
# Launch VQ-VAE hyperparameter sweep jobs on Derecho.
#
# Usage:
#   ./scripts/launch_sweep.sh <config> [config ...]   Submit specific configs
#   ./scripts/launch_sweep.sh sweep1                   Submit all Sweep 1 (scale) configs
#   ./scripts/launch_sweep.sh sweep2                   Submit all Sweep 2 (codebook) configs
#   ./scripts/launch_sweep.sh sweep3                   Submit all Sweep 3 (training) configs
#   ./scripts/launch_sweep.sh --dry-run <config>       Print qsub commands without submitting
#   ./scripts/launch_sweep.sh --list                   List all available configs
#
# Individual configs: A B C D E F  (Sweep 1: scale configs)
#                     V128 V256 V1024 D90 D95 D99 CD32 CD128  (Sweep 2: codebook)
#                     CW01 CW025 CW10 LR1 LR2  (Sweep 3: training dynamics)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DRY_RUN=false

# ---------- Baseline defaults (shared across sweeps) ----------
DEFAULT_SCALES="1x1,2x2,3x3,4x4,5x5,6x6,8x8,10x10,12x12,13x13,16x16"
DEFAULT_VOCAB_SIZE=512
DEFAULT_CODEBOOK_DIM=64
DEFAULT_DECAY=0.85
DEFAULT_LR=4e-4
DEFAULT_COMMITMENT_WEIGHT=0.5
DEFAULT_BATCH_SIZE=64
DEFAULT_BASE_CHANNELS=96
DEFAULT_CHANNEL_MULT="2,4,8,16"
DEFAULT_NUM_RES_BLOCKS=2
DEFAULT_EPOCHS=100
DEFAULT_SAMPLE_STOP=64000

# ---------- Config definitions ----------
# Each function prints VAR=VALUE lines. Only overridden params are listed;
# the PBS template fills in defaults for everything else.

config_vars() {
    local name="$1"
    case "${name}" in
        # ===== Sweep 1: Scale configurations =====
        A)
            echo "RUN_NAME=scales-A-sparse-pow2"
            echo "SCALES=1x1,2x2,4x4,8x8,16x16"
            ;;
        B)
            echo "RUN_NAME=scales-B-current"
            echo "SCALES=${DEFAULT_SCALES}"
            ;;
        C)
            echo "RUN_NAME=scales-C-log-spaced"
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,11x11,16x16"
            ;;
        D)
            echo "RUN_NAME=scales-D-dense-to-14"
            echo "SCALES=1x1,2x2,3x3,4x4,5x5,6x6,7x7,8x8,9x9,10x10,11x11,12x12,13x13,14x14"
            ;;
        E)
            echo "RUN_NAME=scales-E-coarse-only"
            echo "SCALES=1x1,2x2,3x3,4x4,5x5,6x6,7x7,8x8"
            ;;
        F)
            echo "RUN_NAME=scales-F-dense-high"
            echo "SCALES=1x1,4x4,8x8,10x10,12x12,14x14,16x16"
            ;;

        # ===== Sweep 2: Codebook parameters =====
        V128)
            echo "RUN_NAME=vocab-128"
            echo "VOCAB_SIZE=128"
            ;;
        V256)
            echo "RUN_NAME=vocab-256"
            echo "VOCAB_SIZE=256"
            ;;
        V1024)
            echo "RUN_NAME=vocab-1024"
            echo "VOCAB_SIZE=1024"
            ;;
        D90)
            echo "RUN_NAME=decay-0.90"
            echo "DECAY=0.90"
            ;;
        D95)
            echo "RUN_NAME=decay-0.95"
            echo "DECAY=0.95"
            ;;
        D99)
            echo "RUN_NAME=decay-0.99"
            echo "DECAY=0.99"
            ;;
        CD32)
            echo "RUN_NAME=codebook-dim-32"
            echo "CODEBOOK_DIM=32"
            ;;
        CD128)
            echo "RUN_NAME=codebook-dim-128"
            echo "CODEBOOK_DIM=128"
            ;;

        # ===== Sweep 3: Training dynamics =====
        CW01)
            echo "RUN_NAME=commit-0.1"
            echo "COMMITMENT_WEIGHT=0.1"
            ;;
        CW025)
            echo "RUN_NAME=commit-0.25"
            echo "COMMITMENT_WEIGHT=0.25"
            ;;
        CW10)
            echo "RUN_NAME=commit-1.0"
            echo "COMMITMENT_WEIGHT=1.0"
            ;;
        LR1)
            echo "RUN_NAME=lr-1e-4"
            echo "LR=1e-4"
            ;;
        LR2)
            echo "RUN_NAME=lr-2e-4"
            echo "LR=2e-4"
            ;;

        *)
            echo "ERROR: Unknown config '${name}'" >&2
            return 1
            ;;
    esac
}

# ---------- Sweep group definitions ----------
SWEEP1_CONFIGS="A C D E F"     # B (baseline) already running
SWEEP2_CONFIGS="V128 V256 V1024 D90 D95 D99 CD32 CD128"
SWEEP3_CONFIGS="CW01 CW025 CW10 LR1 LR2"
ALL_CONFIGS="${SWEEP1_CONFIGS} ${SWEEP2_CONFIGS} ${SWEEP3_CONFIGS}"

# ---------- Token counting ----------
count_tokens() {
    local scales="$1"
    local total=0
    IFS=',' read -ra SCALE_ARR <<< "${scales}"
    for s in "${SCALE_ARR[@]}"; do
        local h="${s%%x*}"
        local w="${s##*x}"
        total=$((total + h * w))
    done
    echo "${total}"
}

# ---------- Submit one config ----------
submit_config() {
    local name="$1"
    local vars
    vars="$(config_vars "${name}")" || return 1

    # Extract per-config overrides, falling back to defaults
    local run_name scales vocab_size codebook_dim decay lr commitment_weight
    local batch_size base_channels channel_mult num_res_blocks epochs sample_stop
    run_name="$(echo "${vars}" | grep '^RUN_NAME=' | cut -d= -f2- || true)"
    scales="$(echo "${vars}" | grep '^SCALES=' | cut -d= -f2- || true)"
    vocab_size="$(echo "${vars}" | grep '^VOCAB_SIZE=' | cut -d= -f2- || true)"
    codebook_dim="$(echo "${vars}" | grep '^CODEBOOK_DIM=' | cut -d= -f2- || true)"
    decay="$(echo "${vars}" | grep '^DECAY=' | cut -d= -f2- || true)"
    lr="$(echo "${vars}" | grep '^LR=' | cut -d= -f2- || true)"
    commitment_weight="$(echo "${vars}" | grep '^COMMITMENT_WEIGHT=' | cut -d= -f2- || true)"
    batch_size="$(echo "${vars}" | grep '^BATCH_SIZE=' | cut -d= -f2- || true)"
    base_channels="$(echo "${vars}" | grep '^BASE_CHANNELS=' | cut -d= -f2- || true)"
    channel_mult="$(echo "${vars}" | grep '^CHANNEL_MULT=' | cut -d= -f2- || true)"
    num_res_blocks="$(echo "${vars}" | grep '^NUM_RES_BLOCKS=' | cut -d= -f2- || true)"
    epochs="$(echo "${vars}" | grep '^EPOCHS=' | cut -d= -f2- || true)"
    sample_stop="$(echo "${vars}" | grep '^SAMPLE_STOP=' | cut -d= -f2- || true)"

    : "${scales:=${DEFAULT_SCALES}}"
    : "${vocab_size:=${DEFAULT_VOCAB_SIZE}}"
    : "${codebook_dim:=${DEFAULT_CODEBOOK_DIM}}"
    : "${decay:=${DEFAULT_DECAY}}"
    : "${lr:=${DEFAULT_LR}}"
    : "${commitment_weight:=${DEFAULT_COMMITMENT_WEIGHT}}"
    : "${batch_size:=${DEFAULT_BATCH_SIZE}}"
    : "${base_channels:=${DEFAULT_BASE_CHANNELS}}"
    : "${channel_mult:=${DEFAULT_CHANNEL_MULT}}"
    : "${num_res_blocks:=${DEFAULT_NUM_RES_BLOCKS}}"
    : "${epochs:=${DEFAULT_EPOCHS}}"
    : "${sample_stop:=${DEFAULT_SAMPLE_STOP}}"

    local tokens
    tokens="$(count_tokens "${scales}")"

    # Generate a complete, self-contained PBS script.
    # No template stitching — PBS directives are guaranteed at the top.
    local tmpfile
    tmpfile="$(mktemp /tmp/sweep_${name}_XXXXXX.pbs)"
    cat > "${tmpfile}" <<PBSEOF
#!/bin/bash
#PBS -N ${run_name}
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=480GB
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m abe

# ---------- paths ----------
REPODIR="\${HOME}/gust"
DATA_PATH="\${DATA_PATH:-/glade/derecho/scratch/anishs/turb2d_long/output.h5}"
SWEEP_BASE="\${SWEEP_BASE:-/glade/derecho/scratch/anishs/sweep}"
CHECKPOINT_DIR="\${SWEEP_BASE}/${run_name}"

# ---------- training parameters ----------
BASE_CHANNELS="${base_channels}"
CHANNEL_MULT="${channel_mult}"
NUM_RES_BLOCKS="${num_res_blocks}"
SCALES="${scales}"
VOCAB_SIZE="${vocab_size}"
CODEBOOK_DIM="${codebook_dim}"
COMMITMENT_WEIGHT="${commitment_weight}"
EPOCHS="${epochs}"
BATCH_SIZE="${batch_size}"
DECAY="${decay}"
LR="${lr}"
SAMPLE_STOP="${sample_stop}"
WANDB_PROJECT="\${WANDB_PROJECT:-gust-vqvae}"
WANDB_NAME="${run_name}"

# ---------- setup ----------
mkdir -p "\${CHECKPOINT_DIR}"

source /glade/derecho/scratch/anishs/gust-venv/bin/activate

cd "\${REPODIR}"

echo "=========================================="
echo "Job:      \${PBS_JOBID}"
echo "Run:      ${run_name}"
echo "Node:     \$(hostname)"
echo "Started:  \$(date)"
echo "GPUs:     \$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Data:     \${DATA_PATH}"
echo "Samples:  0 - \${SAMPLE_STOP}"
echo "Ckpt dir: \${CHECKPOINT_DIR}"
echo "------------------------------------------"
echo "Scales:           \${SCALES}"
echo "Vocab size:       \${VOCAB_SIZE}"
echo "Codebook dim:     \${CODEBOOK_DIM}"
echo "Decay:            \${DECAY}"
echo "LR:               \${LR}"
echo "Commitment wt:    \${COMMITMENT_WEIGHT}"
echo "Base channels:    \${BASE_CHANNELS}"
echo "Channel mult:     \${CHANNEL_MULT}"
echo "Res blocks:       \${NUM_RES_BLOCKS}"
echo "Batch size:       \${BATCH_SIZE}"
echo "Epochs:           \${EPOCHS}"
echo "=========================================="

# ---------- run ----------
python -m models.train_vqvae \\
    --data_path "\${DATA_PATH}" \\
    --checkpoint_dir "\${CHECKPOINT_DIR}" \\
    --base_channels \${BASE_CHANNELS} \\
    --channel_mult \${CHANNEL_MULT} \\
    --num_res_blocks \${NUM_RES_BLOCKS} \\
    --scales \${SCALES} \\
    --vocab_size \${VOCAB_SIZE} \\
    --codebook_dim \${CODEBOOK_DIM} \\
    --commitment_weight \${COMMITMENT_WEIGHT} \\
    --epochs \${EPOCHS} \\
    --batch_size \${BATCH_SIZE} \\
    --decay \${DECAY} \\
    --lr \${LR} \\
    --sample_stop \${SAMPLE_STOP} \\
    --wandb_project \${WANDB_PROJECT} \\
    --wandb_name \${WANDB_NAME}

echo "=========================================="
echo "Finished: \$(date)"
echo "=========================================="
PBSEOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${name} (${run_name}, ${tokens} tokens):"
        echo "  ${vars//$'\n'/ | }"
        echo "  qsub ${tmpfile}"
        echo "--- generated script ---"
        cat "${tmpfile}"
        echo "--- end ---"
        echo ""
        rm -f "${tmpfile}"
    else
        echo "Submitting ${name} (${run_name}, ${tokens} tokens)..."
        qsub "${tmpfile}"
        rm -f "${tmpfile}"
    fi
}

# ---------- List configs ----------
list_configs() {
    echo "Sweep 1 — Scale configurations:"
    echo "  A     Sparse (pow2):  1,2,4,8,16                                    (341 tokens, 5 scales)"
    echo "  B     Current:        1,2,3,4,5,6,8,10,12,13,16                     (824 tokens, 11 scales) [baseline]"
    echo "  C     Log-spaced:     1,2,3,4,6,8,11,16                             (507 tokens, 8 scales)"
    echo "  D     Dense to 14:    1,2,3,4,5,6,7,8,9,10,11,12,13,14             (1015 tokens, 14 scales)"
    echo "  E     Coarse-only:    1,2,3,4,5,6,7,8                               (204 tokens, 8 scales)"
    echo "  F     Dense-high:     1,4,8,10,12,14,16                              (777 tokens, 7 scales)"
    echo ""
    echo "Sweep 2 — Codebook parameters (run on best scale config from Sweep 1):"
    echo "  V128  vocab_size=128"
    echo "  V256  vocab_size=256"
    echo "  V1024 vocab_size=1024"
    echo "  D90   decay=0.90"
    echo "  D95   decay=0.95"
    echo "  D99   decay=0.99"
    echo "  CD32  codebook_dim=32"
    echo "  CD128 codebook_dim=128"
    echo ""
    echo "Sweep 3 — Training dynamics:"
    echo "  CW01  commitment_weight=0.1"
    echo "  CW025 commitment_weight=0.25"
    echo "  CW10  commitment_weight=1.0"
    echo "  LR1   lr=1e-4"
    echo "  LR2   lr=2e-4"
}

# ---------- Usage ----------
usage() {
    echo "Usage: $0 [--dry-run] [--list] <config|sweep1|sweep2|sweep3> [...]"
    echo ""
    echo "Options:"
    echo "  --dry-run   Print qsub commands without submitting"
    echo "  --list      List all available configurations"
    echo ""
    echo "Examples:"
    echo "  $0 A C D                 Submit scale configs A, C, D"
    echo "  $0 sweep1                Submit all Sweep 1 (scale) configs"
    echo "  $0 --dry-run sweep2      Dry-run all Sweep 2 (codebook) configs"
    echo "  $0 V128 D95              Submit specific codebook configs"
}

# ---------- Main ----------
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

# Parse flags
CONFIGS=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN=true
            ;;
        --list)
            list_configs
            exit 0
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        sweep1)
            CONFIGS+=(${SWEEP1_CONFIGS})
            ;;
        sweep2)
            CONFIGS+=(${SWEEP2_CONFIGS})
            ;;
        sweep3)
            CONFIGS+=(${SWEEP3_CONFIGS})
            ;;
        all)
            CONFIGS+=(${ALL_CONFIGS})
            ;;
        *)
            CONFIGS+=("${arg}")
            ;;
    esac
done

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No configs specified."
    usage
    exit 1
fi

echo "=========================================="
echo "VQ-VAE Hyperparameter Sweep"
echo "Configs: ${CONFIGS[*]}"
echo "Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

for config in "${CONFIGS[@]}"; do
    submit_config "${config}"
done

echo "Done. Submitted ${#CONFIGS[@]} job(s)."
