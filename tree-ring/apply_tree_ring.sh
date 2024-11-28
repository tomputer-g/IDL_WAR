#!/bin/bash

HF_DATASET="/data/CMU/val2017"
SPLIT="validation"
OUTPUT_FOLDER="outputs"
CHANNEL=0
NUM_FILES_TO_PROCESS=-1
RESUME=true
MODEL="stabilityai/stable-diffusion-2-1-base"
SCHEDULER="DPMSolverMultistepScheduler"

python generate_images.py \
    --hf_dataset "$HF_DATASET" \
    --split "$SPLIT" \
    --output_folder "$OUTPUT_FOLDER" \
    --channel "$CHANNEL" \
    --num_files_to_process "$NUM_FILES_TO_PROCESS" \
    --resume "$RESUME" \
    --model "$MODEL" \
    --scheduler "$SCHEDULER"
