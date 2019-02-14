#!/bin/bash

#SBATCH --job-name=fimo_job
#SBATCH --array=0-422
#SBATCH --output=logs/fimo_log_%A_%a.out
#SBATCH --ntasks=1

BG_FILE="/home/zjlin/human_motif/bg_model_human_promoter.txt"
PROMOTER="/home/zjlin/human_motif/Homo_sapiens.GRCh37.75.promoter.-2000.499.fasta"
OUTPUT_DIR="/home/zjlin/human_motif/fimo_result"
MATRICES=(/home/zjlin/human_motif/motif_meme/*.meme)
MATRIX_ID=$(basename -s .meme ${MATRICES[$SLURM_ARRAY_TASK_ID]})
fimo --bfile $BG_FILE --max-stored-scores 10000000 --thresh 1e-3 --o ${OUTPUT_DIR}/$MATRIX_ID ${MATRICES[$SLURM_ARRAY_TASK_ID]} $PROMOTER
