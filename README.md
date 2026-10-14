# Fibcon (muscle fiber spatial connectivity)

## Workflow overview:
1. Collect images of entire muscle cross sections. Until step 4, all image
files are treated independently. Image files should contain metadata that
describes the resolution of the images for downstream analysis. You will have
to correct the measurements if the metadata is not found in step 4. 
3. Use a segmentation algorithm to segment myofibers in the muscle images. 

> **_NOTE_**: Fibcon was developed with cellpose3. Cellpose4 will likely 
produce better segmentation, but requires a GPU to run in a reasonable time. 
The finetuned cellpose3 model for laminin myofiber segmentation can not be used
in cellpose4. 

4. Use provided code (`make_graph_from_img_DT.py`) to collect measurements from
segmented images, build the connectivity graph, and package all that into an
Anndata object for later use.
5. Analyse a directory of Anndata objects using LISA and other spatial data
analysis techniques. 

## Data structure convention
- Image data (one or multiple files per muscle cross section) should all be
collected into a single 'project' directory. Here's a directory structure:

```
RBT                     (the base project directory, containing everything)
├── original_images     (all the images)
├── fiber_masks         (the segmented fiber masks)
├── nuclei_masks        (optional, the segmented nuclei masks)
├── output              (the Anndata files are saved here)
└── log                 (a seperate dir for the log files for debuging)

(scripts & notebooks can also be located in this folder to keep everything together) 
```
- The `nuclei_masks` folder is optional if you have segmented the nuclei. The 
Fibcon pipeline doesn't rely on nuclei, and only looks at myofiber size. Other
variables such as centrally/peripherially located nuclei, myofiber orientation, 
myofiber shape, etc. could be informative to further study spatial relationships
of myofiber heterogeneity. The code for attributing nuclei to their myofibers
is included, but is optional. 

> **_NOTE_**: The nuclei association steps are very time consuming because they
have to iterate over each nuclei and check if they overlap with a myofiber. This
could be algorithmically optimized, and is currently implemented in a naieve way 
but is sped up with multiprocessing, see the `--nthreads` flag in `make_graph_from_img_DT.py`. 

## Installation
- Installation should work with either conda or venv using `requirements.txt`. 
Below are instructions for using venv.
- Create a venv environment with python: 
```bash
python -m venv venv
```

This creates a virtual environment named `venv` that isolates the dependencies 
used in this project. The environment is a folder that contains scripts to 
activate the environment.

- Activate the environment on Windows by navigating to to the `venv/Scripts/` dir
and running `activate.bat` or on a Unix system by running `./venv/bin/activate`
in the terminal. 
- Finally, install the required packages into the environment: `pip install -r requirements.txt`

## Rachel's RBT bioprinting data notes:
- Make fiber masks for entire folder:

```bash
for f in ./*; do sbatch fiber_seg_CLI.slurm "$f"; done
```

- Make nuclei masks for entire folder:

```bash
for f in ./*; do sbatch run_borked_nuclei_seg.slurm "$f"; done
```

