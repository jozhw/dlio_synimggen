#!/bin/bash

# Start timer
start_time=$(date +%s.%N)

# define a constant for number of image paths to get
NUM_IMG_PATHS=100

# imagenet path
IMAGENET_PATH="eagle/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train"

# Change directory directly instead of using multiple 'cd' commands
IMAGENET_DIR="$HOME/../../$IMAGENET_PATH"

# Check if the directory exists
if [ ! -d "$IMAGENET_DIR" ]; then
    echo "Could not find $IMAGENET_PATH"
    exit 1
fi

# Find image paths given NUM_IMG_PATHS 
jpg_files=$(find "$IMAGENET_DIR" -type f -name '*.JPEG' | shuf -n "$NUM_IMG_PATHS")

# Generate JSON file name
json_file="imagenet_rand_$NUM_IMG_PATHS.json"

# Create directory if it doesn't exist
output_dir="assets/polaris/img_paths/$(date +'%Y-%m-%d')"
mkdir -p "$output_dir"

# Store paths in a temporary file
tmp_file=$(mktemp)
for file in $jpg_files; do
    realpath "$file"
done > "$tmp_file"

# create json file containing all of the paths
jq -Rs '{paths: split("\n") | map(select(. != ""))}' "$tmp_file" > "$output_dir/$json_file"

# Remove the temporary file
rm "$tmp_file"

# End timer
end_time=$(date +%s.%N)

# Calculate elapsed time
execution_time=$(echo "$end_time - $start_time" | bc)

# Print execution time
echo "Execution time: $execution_time seconds for $NUM_IMG_PATHS image paths."
