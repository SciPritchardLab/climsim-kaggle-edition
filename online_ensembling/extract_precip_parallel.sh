#!/bin/bash
#SBATCH --job-name=extract_precip_parallel
#SBATCH -A m4334
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 30:00
#SBATCH --output=extract_precip_parallel_%j.out
#SBATCH --error=extract_precip_parallel_%j.err
#SBATCH -c 64
#SBATCH --mail-user=jerryL9@uci.edu
#SBATCH --mail-type=ALL

# Check if filepath argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No filepath provided. Usage: sbatch $0 /path/to/directory"
    exit 1
fi

# Change to the specified directory
echo "Changing to directory: $1"
cd "$1" || { echo "Error: Failed to change to directory $1"; exit 1; }

module load conda
conda activate plotting

outdir="precip_dir"
export outdir # <-- Add this line

# Remove precip_dir if it already exists
if [ -d "$outdir" ]; then
    echo "Removing existing $outdir directory"
    rm -rf "$outdir"
fi

mkdir -p "$outdir"

# Function to extract PRECC and PRECT variables from a single file
extract_precip() {
    file="$1"
    output_file="${outdir}/precip_${file##*/}"  # Use ##*/ to remove directory path
    ncks -v PRECC,PRECT "$file" "$output_file"
    echo "Extracted PRECC and PRECT from $file to $output_file"
}

# Export the function for GNU Parallel
export -f extract_precip

# Use find to generate the list of files and pipe it to GNU Parallel
find . -maxdepth 1 -type f -name "*_seed_*.eam.h2.000[34567]-*.nc" |
  parallel -j 64 extract_precip {}

combined_file="${outdir}/combined_precip.nc"

# Collect files into a Bash array
mapfile -d '' files < <(find "${outdir}" -maxdepth 1 -type f -name "precip_*.nc" -print0 | sort -zV)

if [ ${#files[@]} -eq 0 ]; then
    echo "Error: No files matching 'precip_*.nc' found in ${outdir}"
    exit 1
fi

# ----------------------------------------------------
# NEW: Process files in batches to avoid "arg list too long"
# ----------------------------------------------------
tmp_prefix="${outdir}/tmp_combine_"
batch_size=500  # Adjust based on your system's argument limit

# Create temporary files for each batch
for ((i=0; i<${#files[@]}; i+=$batch_size)); do
    batch_files=(${files[@]:i:batch_size})  # Extract a batch of files
    printf -v pad "%05d" $i
    tmp_file="${tmp_prefix}${pad}.nc"
    ncrcat -O -o "${tmp_file}" "${batch_files[@]}" && \
    echo "Created batch ${tmp_file}"
done

# Combine all temporary files into the final output
printf '%s\0' "${outdir}/tmp_combine_"*.nc | sort -zV | xargs -0 ncrcat -C -O -o "${combined_file}"
# ncrcat -O -o "${combined_file}" ${outdir}/tmp_combine_*.nc

# Clean up temporary files
rm ${outdir}/tmp_combine_*.nc