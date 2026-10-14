#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --output=result_%a.out
#SBATCH --partition=standard
#SBATCH --mem=64Gb
#SBATCH --cpus-per-task=14
#SBATCH -A christlab

module purge
module load gcc/16.2.0
module load python/3.14.7
source /scratch/hyt6ex/fibcon-main/venv/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /scratch/hyt6ex/RBT || exit 1


# Get files
mapfile -t filelist < <(ls /scratch/hyt6ex/RBT/fiber_masks/*png)
file="${filelist[${SLURM_ARRAY_TASK_ID}]}"
base=$(basename "$file")
id="${base%%_cp_masks*}"
img="/scratch/hyt6ex/RBT/original_images/${id}.tif"
out="/scratch/hyt6ex/RBT/output/${base%.*}_adata.h5ad"
if [ -f "$out" ]; then
  echo "Skipping $id (output already exists)"
  exit 0
fi


echo Processing $file
python /scratch/hyt6ex/fibcon-main/make_graph_from_img_DT.py \
  --fiber_mask_path "$file" \
  --original_image_path "$img" \
  --resolution 0.65


