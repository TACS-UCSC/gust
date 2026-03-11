#!/bin/bash
# Launch tokenizer (VQ-VAE) sweep: 6 configs (3 scales x 2 channels),
# each chain-submitted as multiple PBS jobs for long training.
#
# Usage:
#   ./scripts/launch_tokenizer_sweep.sh                       Submit all 6 configs (3 chained jobs each)
#   ./scripts/launch_tokenizer_sweep.sh regular-S light-M     Submit specific configs
#   ./scripts/launch_tokenizer_sweep.sh --jobs 5 regular-S    Override chain length
#   ./scripts/launch_tokenizer_sweep.sh --dry-run             Print without submitting
#   ./scripts/launch_tokenizer_sweep.sh --list                List configs

set -euo pipefail

DRY_RUN=false
NJOBS=3

SCALE_CONFIGS="regular-S regular-M light-S light-M full-S full-M"
CODEBOOK_CONFIGS="cb-V128 cb-V256 cb-V1024 cb-D70 cb-D95 cb-D99 cb-CD32 cb-CD128"
VOCAB_CONFIGS="S-V512 S-V1024 S-V2048 M-V512 M-V1024 M-V2048"
ALL_CONFIGS="${SCALE_CONFIGS} ${CODEBOOK_CONFIGS} ${VOCAB_CONFIGS}"

# ---------- Config definitions ----------
config_vars() {
    local name="$1"
    case "${name}" in
        regular-S)
            echo "SCALES=1x1,2x2,4x4,8x8,16x16"
            echo "BASE_CHANNELS=64"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            ;;
        regular-M)
            echo "SCALES=1x1,2x2,4x4,8x8,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            ;;
        light-S)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=64"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            ;;
        light-M)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            ;;
        full-S)
            echo "SCALES=1x1,2x2,3x3,4x4,5x5,6x6,8x8,10x10,12x12,13x13,16x16"
            echo "BASE_CHANNELS=64"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            ;;
        full-M)
            echo "SCALES=1x1,2x2,3x3,4x4,5x5,6x6,8x8,10x10,12x12,13x13,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            ;;
        # ===== Codebook sweep (light-M base) =====
        cb-V128)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "VOCAB_SIZE=128"
            echo "WANDB_GROUP=codebook-sweep"
            ;;
        cb-V256)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "VOCAB_SIZE=256"
            echo "WANDB_GROUP=codebook-sweep"
            ;;
        cb-V1024)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "VOCAB_SIZE=1024"
            echo "WANDB_GROUP=codebook-sweep"
            ;;
        cb-D70)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "DECAY=0.70"
            echo "WANDB_GROUP=codebook-sweep"
            ;;
        cb-D95)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "DECAY=0.95"
            echo "WANDB_GROUP=codebook-sweep"
            ;;
        cb-D99)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "DECAY=0.99"
            echo "WANDB_GROUP=codebook-sweep"
            ;;
        cb-CD32)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "CODEBOOK_DIM=32"
            echo "WANDB_GROUP=codebook-sweep"
            ;;
        cb-CD128)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "CODEBOOK_DIM=128"
            echo "WANDB_GROUP=codebook-sweep"
            ;;

        # ===== Vocab x channels sweep (light scales) =====
        S-V512)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=64"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "VOCAB_SIZE=512"
            echo "WANDB_GROUP=vocab-sweep"
            ;;
        S-V1024)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=64"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "VOCAB_SIZE=1024"
            echo "WANDB_GROUP=vocab-sweep"
            ;;
        S-V2048)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=64"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "VOCAB_SIZE=2048"
            echo "WANDB_GROUP=vocab-sweep"
            ;;
        M-V512)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "VOCAB_SIZE=512"
            echo "WANDB_GROUP=vocab-sweep"
            ;;
        M-V1024)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "VOCAB_SIZE=1024"
            echo "WANDB_GROUP=vocab-sweep"
            ;;
        M-V2048)
            echo "SCALES=1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16"
            echo "BASE_CHANNELS=80"
            echo "CHANNEL_MULT=2,4,8,16"
            echo "NUM_RES_BLOCKS=2"
            echo "VOCAB_SIZE=2048"
            echo "WANDB_GROUP=vocab-sweep"
            ;;

        *)
            echo "ERROR: Unknown config '${name}'" >&2
            return 1
            ;;
    esac
}

# ---------- Submit one config (generate PBS + chain-submit) ----------
submit_config() {
    local name="$1"
    local vars
    vars="$(config_vars "${name}")" || return 1

    # Extract per-config values
    local scales base_channels channel_mult num_res_blocks
    local config_vocab_size config_codebook_dim config_decay config_wandb_group
    scales="$(echo "${vars}" | grep '^SCALES=' | cut -d= -f2-)"
    base_channels="$(echo "${vars}" | grep '^BASE_CHANNELS=' | cut -d= -f2-)"
    channel_mult="$(echo "${vars}" | grep '^CHANNEL_MULT=' | cut -d= -f2-)"
    num_res_blocks="$(echo "${vars}" | grep '^NUM_RES_BLOCKS=' | cut -d= -f2-)"
    config_vocab_size="$(echo "${vars}" | grep '^VOCAB_SIZE=' | cut -d= -f2- || true)"
    config_codebook_dim="$(echo "${vars}" | grep '^CODEBOOK_DIM=' | cut -d= -f2- || true)"
    config_decay="$(echo "${vars}" | grep '^DECAY=' | cut -d= -f2- || true)"
    config_wandb_group="$(echo "${vars}" | grep '^WANDB_GROUP=' | cut -d= -f2- || true)"

    # Non-swept defaults (overridable per config)
    local vocab_size="${config_vocab_size:-512}"
    local codebook_dim="${config_codebook_dim:-64}"
    local decay="${config_decay:-0.85}"
    local wandb_group="${config_wandb_group:-}"
    local lr=4e-4
    local commitment_weight=0.5
    local batch_size=64
    local epochs=250
    local sample_stop=64000

    # Generate a unique wandb ID for this config's chain
    local wandb_id="tok-${name}-$(head -c 4 /dev/urandom | od -An -tx1 | tr -d ' \n')"

    # Generate self-contained PBS script
    local tmpfile
    tmpfile="$(mktemp /tmp/tok_${name}_XXXXXX.pbs)"
    cat > "${tmpfile}" <<PBSEOF
#!/bin/bash
#PBS -N tok-${name}
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=480GB
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m abe

# ---------- paths ----------
REPODIR="\${HOME}/gust"
DATA_PATH="\${DATA_PATH:-/glade/derecho/scratch/anishs/turb2d_long/output.h5}"
SWEEP_BASE="\${SWEEP_BASE:-/glade/derecho/scratch/anishs/${wandb_group:-tokenizer-sweep}}"
CHECKPOINT_DIR="\${SWEEP_BASE}/${name}"

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
WANDB_PROJECT="\${WANDB_PROJECT:-gust-tokenizer-sweep}"
WANDB_NAME="${name}"

# ---------- chaining support ----------
# WANDB_ID is passed via qsub -v for multi-job run continuation.
# Auto-detect resume: if a prior checkpoint exists, add --resume.
RESUME_FLAG=""
if [ -f "\${CHECKPOINT_DIR}/training_state.json" ]; then
    RESUME_FLAG="--resume"
    echo "Found prior checkpoint -- will resume training"
fi

WANDB_FLAGS=""
if [ -n "\${WANDB_ID:-}" ]; then
    WANDB_FLAGS="--wandb_id \${WANDB_ID} --wandb_name \${WANDB_ID}"
fi

WANDB_GROUP_FLAG="${wandb_group:+--wandb_group ${wandb_group}}"

# ---------- setup ----------
mkdir -p "\${CHECKPOINT_DIR}"

# Write wandb logs and temp files to scratch (not home, which has a small quota)
export WANDB_DIR="/glade/derecho/scratch/anishs/wandb"
export TMPDIR="/glade/derecho/scratch/anishs/tmp"
mkdir -p "\${WANDB_DIR}" "\${TMPDIR}"

source /glade/derecho/scratch/anishs/gust-venv/bin/activate

cd "\${REPODIR}"

echo "=========================================="
echo "Job:      \${PBS_JOBID}"
echo "Run:      ${name}"
echo "Node:     \$(hostname)"
echo "Started:  \$(date)"
echo "GPUs:     \$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Data:     \${DATA_PATH}"
echo "Samples:  0 - \${SAMPLE_STOP}"
echo "Ckpt dir: \${CHECKPOINT_DIR}"
echo "Wandb ID: \${WANDB_ID:-<new run>}"
echo "Resume:   \${RESUME_FLAG:-no}"
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
    \${RESUME_FLAG} \\
    \${WANDB_FLAGS} \\
    \${WANDB_GROUP_FLAG}

echo "=========================================="
echo "Finished: \$(date)"
echo "=========================================="
PBSEOF

    # Chain-submit NJOBS copies with depend=afterany
    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${name} (WANDB_ID=${wandb_id}, ${NJOBS} chained jobs):"
        echo "  scales=${scales}  base_channels=${base_channels}"
        echo "--- generated script ---"
        cat "${tmpfile}"
        echo "--- end ---"
        echo ""
        rm -f "${tmpfile}"
        return
    fi

    echo "Submitting ${name} (${NJOBS} chained jobs, WANDB_ID=${wandb_id})..."

    local prev_jobid=""
    for i in $(seq 1 "${NJOBS}"); do
        local qsub_args=(-v "WANDB_ID=${wandb_id}")

        if [ -n "${prev_jobid}" ]; then
            qsub_args+=(-W "depend=afterany:${prev_jobid}")
        fi

        local jobid
        jobid=$(qsub "${qsub_args[@]}" "${tmpfile}")
        echo "  Job ${i}/${NJOBS}: ${jobid}"
        prev_jobid="${jobid}"
    done

    rm -f "${tmpfile}"
    echo ""
}

# ---------- List configs ----------
list_configs() {
    echo "Scale sweep: 3 scale configs x 2 channel configs = 6 runs"
    echo ""
    echo "Scale configs:"
    echo "  regular   1x1,2x2,4x4,8x8,16x16                                    (5 scales)"
    echo "  light     1x1,2x2,3x3,4x4,6x6,8x8,12x12,16x16                     (8 scales)"
    echo "  full      1x1,2x2,3x3,4x4,5x5,6x6,8x8,10x10,12x12,13x13,16x16    (11 scales)"
    echo ""
    echo "Channel configs:"
    echo "  S         base_channels=64, channel_mult=2,4,8,16, num_res_blocks=2"
    echo "  M         base_channels=80, channel_mult=2,4,8,16, num_res_blocks=2"
    echo ""
    echo "Run names:"
    echo "  regular-S  regular-M  light-S  light-M  full-S  full-M"
    echo ""
    echo "Codebook sweep (light-M base, group=codebook-sweep): 8 runs"
    echo "  cb-V128   vocab_size=128"
    echo "  cb-V256   vocab_size=256"
    echo "  cb-V1024  vocab_size=1024"
    echo "  cb-D70    decay=0.70"
    echo "  cb-D95    decay=0.95"
    echo "  cb-D99    decay=0.99"
    echo "  cb-CD32   codebook_dim=32"
    echo "  cb-CD128  codebook_dim=128"
    echo ""
    echo "Vocab x channels sweep (light scales, group=vocab-sweep): 6 runs"
    echo "  S-V512    base_channels=64, vocab_size=512"
    echo "  S-V1024   base_channels=64, vocab_size=1024"
    echo "  S-V2048   base_channels=64, vocab_size=2048"
    echo "  M-V512    base_channels=80, vocab_size=512"
    echo "  M-V1024   base_channels=80, vocab_size=1024"
    echo "  M-V2048   base_channels=80, vocab_size=2048"
}

# ---------- Usage ----------
usage() {
    echo "Usage: $0 [--dry-run] [--jobs N] [--list] [config ...]"
    echo ""
    echo "Options:"
    echo "  --dry-run      Print generated PBS scripts without submitting"
    echo "  --jobs N       Number of chained jobs per config (default: 3)"
    echo "  --list         List all available configurations"
    echo ""
    echo "If no configs specified, submits all scale configs."
    echo "Groups: scales, codebook, vocab, all"
    echo ""
    echo "Examples:"
    echo "  $0                              Submit all scale configs (3 chained jobs each)"
    echo "  $0 codebook                     Submit all codebook sweep configs"
    echo "  $0 regular-S light-M            Submit specific configs"
    echo "  $0 --jobs 5 regular-S           Override chain length"
    echo "  $0 --dry-run codebook           Dry-run codebook sweep"
}

# ---------- Main ----------
CONFIGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --jobs)
            NJOBS="${2:?--jobs requires a number}"
            shift 2
            ;;
        --list)
            list_configs
            exit 0
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        scales)
            read -ra _tmp <<< "${SCALE_CONFIGS}"
            CONFIGS+=("${_tmp[@]}")
            shift
            ;;
        codebook)
            read -ra _tmp <<< "${CODEBOOK_CONFIGS}"
            CONFIGS+=("${_tmp[@]}")
            shift
            ;;
        vocab)
            read -ra _tmp <<< "${VOCAB_CONFIGS}"
            CONFIGS+=("${_tmp[@]}")
            shift
            ;;
        all)
            read -ra _tmp <<< "${ALL_CONFIGS}"
            CONFIGS+=("${_tmp[@]}")
            shift
            ;;
        *)
            CONFIGS+=("$1")
            shift
            ;;
    esac
done

# Default: scale configs only
if [ ${#CONFIGS[@]} -eq 0 ]; then
    read -ra CONFIGS <<< "${SCALE_CONFIGS}"
fi

echo "=========================================="
echo "Tokenizer Sweep"
echo "Configs: ${CONFIGS[*]}"
echo "Jobs per config: ${NJOBS}"
echo "Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

for config in "${CONFIGS[@]}"; do
    submit_config "${config}"
done

echo "Done. Submitted ${#CONFIGS[@]} config(s) x ${NJOBS} jobs = $((${#CONFIGS[@]} * NJOBS)) total jobs."
