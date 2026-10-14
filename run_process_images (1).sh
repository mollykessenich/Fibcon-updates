#!/bin/bash

mapfile -t files < <(ls /scratch/hyt6ex/RBT/fiber_masks/*png)
numfiles="${#files[@]}"
echo Processing $numfiles files
sbatch --array=0-$((numfiles - 1)) /scratch/hyt6ex/fibcon-main/slurm/process_images.slurm
