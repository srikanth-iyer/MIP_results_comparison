#!/bin/bash

# Remote server details
REMOTE_USER="gs5183"
REMOTE_HOST="della.princeton.edu"
REMOTE_DIR="/scratch/gpfs/gs5183/MIP/op_cases"

# Get the current directory name
CURRENT_FOLDER=$(basename "$PWD")
PARENT_DIR="$(dirname "$dir")"

# Append the current folder name to the remote directory path
REMOTE_DIR="$REMOTE_DIR/$CURRENT_FOLDER"

FILES=( "capacityfactor.csv" "costs.csv" "nse.csv")

for FILE in "${FILES[@]}"; do
    rsync -vr --include="${FILE}" --exclude="*" "$REMOTE_USER@$REMOTE_HOST:${REMOTE_DIR}/" .
done

# MODELS=( "SWITCH" "GenX" "TEMOA" ) # "TEMOA" "SWITCH" "GenX" "USENSYS"

# # Local destination directories
# LOCAL_DIR_CAPACITY="/path/to/local/destination/capacityfactor"
# LOCAL_DIR_COSTS="/path/to/local/destination/costs"
# LOCAL_DIR_NSE="/path/to/local/destination/nse"

# # Create local directories if they don't exist
# mkdir -p "$LOCAL_DIR_CAPACITY"
# mkdir -p "$LOCAL_DIR_COSTS"
# mkdir -p "$LOCAL_DIR_NSE"

# FILES=( "capacityfactor.csv" "costs.csv" "nse.csv")

# for MODEL in "${MODELS[@]}"; do
#     echo "${MODEL}"
#     for i in {1..6}; do
#         echo "p${i}"
#         for FILE in "${FILES[@]}"; do
#             DEST="${MODEL}_op_inputs/Inputs/Inputs_p${i}/Results"
#             mkdir -p "${PWD}/${DEST}"
#             rsync -avz --dry-run --include="*/" --include="${FILE}" --exclude="*" \
#     "$REMOTE_USER@$REMOTE_HOST:${REMOTE_DIR}/${DEST}/" "${PWD}/${DEST}/"
#             # scp "$REMOTE_USER@$REMOTE_HOST:${REMOTE_DIR}/${DEST}/${FILE}" "${PWD}/${DEST}"
#         done
#     done
# done

# # Search and copy capacityfactor.csv
# rsync -avz --include="*/" --include="capacityfactor.csv" --exclude="*" \
#     "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR_CAPACITY/"

# # Search and copy costs.csv
# rsync -avz --include="*/" --include="costs.csv" --exclude="*" \
#     "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR_COSTS/"

# # Search and copy nse.csv
# rsync -avz --include="*/" --include="nse.csv" --exclude="*" \
#     "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR_NSE/"

echo "Files have been copied successfully."
