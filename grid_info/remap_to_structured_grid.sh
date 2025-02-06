#!/bin/bash

module load conda
conda activate yourenv
for infile in your_run.eam.h2*.nc; do
    outfile="${infile/your_run/remapped_run}"
    ncks --map=/your/folder/here/map_ne4pg2_to_180x360_lowres.nc "$infile" "$outfile"
done
