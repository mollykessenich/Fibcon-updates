Fibcon on AFIRM3 Aim 1c GN Histology data: fixes and setup
Overview/the pipeline: 
1.	Masks and images. RBT/fiber_masks/ holds 66 Cellpose masks (as of now, BP/CG/NR groups & missing some being re-stained); RBT/original_images/ holds the matching .tif images.
2.	Graph building. make_graph_from_img_DT.py runs once per mask through a Slurm array job (run_process_images.sh submits process_images.slurm). It measures each fiber, builds the Delaunay neighbor graph, runs LISA on minFD and eccentricity, and computes message-passing clusters.
3.	Output. One .h5ad file per section in RBT/output/, named after its mask.
4.	Analysis. BAA_LAD_analysis-adata.ipynb loads the .h5ad files and compares groups/creates visual comparisons.
CHANGES MADE/ADDITIONS
1. utils.py
File: /scratch/hyt6ex/fibcon-main/utils.py. 
In calc_lisa:
Name	Library	What it does
W	libpysal	Builds the spatial weights object: the record of which fiber neighbors which
lag_spatial	libpysal	For each fiber, the average value of its neighbors (for example, their mean minFD)
Moran	esda	Global Moran’s I: one number for the whole section saying whether similar-sized fibers sit together
Moran_Local	esda	Local Moran’s I (LISA): one result per fiber, which is where HH / LL / HL / LH come from
Fix 1: missing imports. calculate_LISA uses W, Moran, Moran_Local and lag_spatial, but nothing imported them. libpysal builds the spatial-weights object (which fiber neighbors which); esda computes global Moran's I and local LISA on top of it. * added two lines at top of file after other imports → fixed NameError: name 'W' is not defined
from libpysal.weights import W, lag_spatial
from esda.moran import Moran, Moran_Local

Fix 2: get_graph_W. The line (~ 625) had been changed to call get_graph_W itself, but the original W fxn is used to create it  
def get_graph_W(in_graph):
    this_graph_dict = {}
    for node in in_graph.nodes():
        this_graph_dict[node] = list(in_graph.neighbors(node))

    this_graph_w = W(neighbors=this_graph_dict)
    return this_graph_w

^W fxn actually from library, but the get_graph_W preps data so it can be used (w = spatial weights, so it stores neighbors as a networkx graph, built from the Delaunay triangulation: fibers = nodes, fibers that share a border = connected by edges. get_graph_W converts one format into the other)
Fix 3: read-only array. Newer pandas returns a read-only view, and the next lines subtract the mean in place. .copy() makes an editable copy; the values are unchanged.
feature_arr = adata[:, [feature_name]].to_df().values.copy()
*venv environment uses newer pandas, changes nothing abt values only makes them editable/change who “owns” dataframe (other environment returned writable arrays so worked but updated to match environment needed the .copy() added)
Fix 4: speed-up. Each fiber checked node in subset_nodes against a NumPy array, which scans the whole array every time. The ns group alone took about 10 minutes per section. A Python set makes each check nearly instant.


# before
def get_subset_distance_distribution(subset_name, subset_nodes, G):
    node_dists = {}
    for node in subset_nodes:
        dists = compute_node_distances(node, subset_nodes, G)
        node_dists[node] = dists
    return subset_name, node_dists

# after
def get_subset_distance_distribution(subset_name, subset_nodes, G):
    subset_set = set(np.asarray(subset_nodes).tolist())[1.1]
    node_dists = {}
    for node in subset_nodes:
        dists = compute_node_distances(node, subset_set, G)
        node_dists[node] = dists
    return subset_name, node_dists

Not fixed, because the pipeline never calls them: threshold_minimum and csv are used without imports, np.int0 no longer exists in NumPy, and are_coords_touching refers to undefined variables.
2. Python environment (venv)
Location: /scratch/hyt6ex/fibcon-main/venv. A venv has no Python of its own; it points back to the Python module that built it, so that module must be loaded first, every time.
Error	Cause	Fix
No module named 'libpysal' / 'esda'	Not in requirements.txt	pip install libpysal esda
libpython3.14.so.1.0: cannot open shared object file	Python module not loaded	Load modules before activating (below)
venv/bin/activate: No such file or directory	The venv folder was deleted or moved on Oct 7 * when messing with jupyter nb 	Rebuilt it (below, how it was rebuilt the first)
Notebook: No module named 'anndata'	Jupyter used its default Python	Registered the venv as a kernel (section 6) →caused issue ^
Gcc, then Python, then activate. The Python module = visible once gcc is loaded, and the venv only runs once Python is loaded. → just changes environment and updated to more recent → caused other changes before but don’t affect the actual data/pipeline just loading python
module load gcc/16.2.0
module load python/3.14.7
source /scratch/hyt6ex/fibcon-main/venv/bin/activate 

# before 
module purge
module load anaconda/2020.11-py3.8
eval "$(conda shell.bash hook)"
source activate /scratch/hyt6ex/.conda_envs/fibcon/
module load gcc
Rebuild steps, if the venv is ever missing again:
module load gcc/16.2.0
module load python/3.14.7
cd /scratch/hyt6ex/fibcon-main
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install libpysal esda ipykernel scanpy seaborn pingouin statannotations splot
python3 -c "import anndata, squidpy, libpysal, esda; print('OK')"

Pairing rule. An original image's name is its mask's name with everything from _cp_masks onward removed, plus .tif. For example, Aim1c_LG_BP01_expt_s1c1-3_cp_masks.png pairs with Aim1c_LG_BP01_expt_s1c1-3.tif. In bash: id="${base%%_cp_masks*}". This also handles the _test and  (1) suffixes.


3. Submitting batch job/running pipeline
**An array job asks Slurm to run the same script many times, each copy with a different number:
-	run_process_images.sh is run once → counts the masks and asks Slurm for that many copies of process_images.slurm. → process_images.slurm = “worker”. Each copy processes one sample. Slurm gives each copy a number in the variable SLURM_ARRAY_TASK_ID: 0, 1, 2, … 65.
-	Each works out sample the same way: list the masks and take item number SLURM_ARRAY_TASK_ID. Task 0 takes the first mask, task 1 the second, and so on.
*both files must build exactly the same list in the same order. The first uses the list to decide # tasks to create, and other uses its own copy of the list to pick a file. If the two lists differ, tasks get the wrong file or no file at all. 
process_images.slurm
File: /scratch/hyt6ex/fibcon-main/slurm/process_images.slurm. Each array task processes one mask. The original = written for an older conda setup and older scripts; these are the minimum changes to make it run.
Fix 1: time limit/environment. --time = max run time for each task. Slurm stops a task that goes over, and nothing is saved. One sample took ~15 min before the speed-up (bigger sections longer), so 30 min was too close. 2 hours = safe margin; you are only charged for time actually used.
# before
#SBATCH --time=0:30:00

# after
#SBATCH --time=2:00:00

FIX ENVIRONMENT  from anaconda to the venv envir.
# before
module purge
module load anaconda/2020.11-py3.8
eval "$(conda shell.bash hook)"
# source activate /scratch/hyt6ex/.conda_envs/fibcon/
source /scratch/hyt6ex/fibcon-main/venv/bin/activate

module load gcc

# after 
module purge
module load gcc/16.2.0
module load python/3.14.7
source /scratch/hyt6ex/fibcon-main/venv/bin/activate


Fix 2: thread limit. Environment lines (module purge → activate) are in section 2. One line was also added after them: OpenBLAS (the math library under numpy) otherwise tries to use every core on the node → “OpenBLAS error: Memory allocation still failed”. This caps it at the CPUs the job asked for.
# before
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# after
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

Fix 3: file list = masks only. ls prints the matching file names (alphabetical, one per line) and mapfile -t reads them into a bash list called filelist. The old line listed images AND masks (136 items: 66 .tif then 66 .png), so tasks 0–65 would each get an image, not a mask. make_graph_from_img_DT.py needs the mask first and finds the image separately (Fix 5), so the list is masks only = 66 items, one per sample.
# before
mapfile -t filelist < <(ls /scratch/hyt6ex/RBT/original_images/*tif /scratch/hyt6ex/RBT/fiber_masks/*png)

# after
mapfile -t filelist < <(ls /scratch/hyt6ex/RBT/fiber_masks/*png)

Fix 4: sort line removed. Looked like it sorted the list but did nothing. ${filelist[*]} squashes the list into one string; the | sort puts the line in a pipeline, which runs in a temporary copy of the shell, so the real filelist was never changed, and sort got no input: ls already returns files in alphabetical order.
# before
filelist=${filelist[*]} | sort

# after
(line removed)

Fix 5: pick this task’s mask + find its image. The first line (unchanged) takes item number SLURM_ARRAY_TASK_ID from the list. The three added lines find the matching original image from the mask’s name (same as the pairing rule in section 2), because the new script needs the image passed in explicitly.
# before
file="${filelist[${SLURM_ARRAY_TASK_ID}]}"

# after
file="${filelist[${SLURM_ARRAY_TASK_ID}]}"
base=$(basename "$file")
id="${base%%_cp_masks*}"
img="/scratch/hyt6ex/RBT/original_images/${id}.tif"

Example, task 3:
Line	Result
file="${filelist[3]}"	/scratch/hyt6ex/RBT/fiber_masks/Aim1c_LG_BP03_expt_s1c1-3_cp_masks.png
base=$(basename "$file") → strips the folder	Aim1c_LG_BP03_expt_s1c1-3_cp_masks.png
id="${base%%_cp_masks*}" → cuts off _cp_masks and everything after	Aim1c_LG_BP03_expt_s1c1-3
img=".../original_images/${id}.tif"	/scratch/hyt6ex/RBT/original_images/Aim1c_LG_BP03_expt_s1c1-3.tif
*%% = bash “remove from the end”: removes the longest ending that starts with _cp_masks → handles _cp_masks.png, _cp_masks_test.png and _cp_masks (1).png the same way.

Fix 6: change path 
# before 
python /scratch/hyt6ex/fibcon-main/make_graph_from_img.py $file
python /scratch/hyt6ex/fibcon-main/make_graph_from_img_5c.py $file

after: (files didn’t exist, so changed to use the one that is in the fibcon-main folder)

# after
python /scratch/hyt6ex/fibcon-main/make_graph_from_img_DT.py \
  --fiber_mask_path "$file" \
  --original_image_path "$img" \
  --resolution 0.65



Unchanged: #SBATCH -A christlab, --partition, --mem, --cpus-per-task, --output=result_%a.out and cd /scratch/hyt6ex/RBT

run_process_images.sh
File: /scratch/hyt6ex/fibcon-main/slurm/run_process_images.sh. The launcher: run once; counts the masks and submits one array task per mask. Changed lines highlighted.
# before
#!/bin/bash

mapfile -t files < <(ls /scratch/hyt6ex/RBT/original_images/*tif /scratch/hyt6ex/RBT/fiber_masks/*png)
numfiles="${#files[@]}"
echo Processing $numfiles files
sbatch --array=0-${numfiles} process_images.slurm

# after
#!/bin/bash

mapfile -t files < <(ls /scratch/hyt6ex/RBT/fiber_masks/*png)
numfiles="${#files[@]}"
echo Processing $numfiles files
sbatch --array=0-$((numfiles - 1)) /scratch/hyt6ex/fibcon-main/slurm/process_images.slurm


Fix 1: list = masks only. Same change as process_images.slurm Fix 3, for the same reason: the two lists must match. Listing images + masks is why it printed “Processing 136 files”.
Fix 2: array range. numfiles="${#files[@]}" counts the list (66). --array=0-${numfiles} = 0-66 = 67 tasks, but list positions only run 0–65 (counting starts at 0), so task 66 would get an empty name. $((numfiles - 1)) is bash arithmetic → 65 → --array=0-65 = exactly 66 tasks, one per mask.
Fix 3: full path to process_images.slurm. A bare file name = “look in the folder I’m in right now”. Run from home or RBT (not slurm/), sbatch couldn’t find it → “Unable to open file process_images.slurm”. A full path works from any folder.


To run (from RBT so the result_N.out logs land there):
cd /scratch/hyt6ex/RBT
bash /scratch/hyt6ex/fibcon-main/slurm/run_process_images.sh

*each task rebuilds its list when it STARTS, not when you submit. If a mask is added, renamed or removed while tasks are still waiting, the alphabetical positions shift and a task can process the wrong mask, or skip/repeat one → leave fiber_masks/ alone until squeue --me shows nothing left.

5.	Analysis notebook and Jupyter kernel
** check before starting if a batch job is running (will interupt and stop the job)
module load gcc/16.2.0
module load python/3.14.7
source /scratch/hyt6ex/fibcon-main/venv/bin/activate
python -c "import ipykernel, scanpy, seaborn, pingouin, statannotations, splot; print('all installed')"

Kernel setup (one time). Without this, notebooks run on Jupyter's default Python and cannot find anndata.
1.	Create ~/fibcon_kernel.sh. It loads the modules, activates the venv, then starts the kernel.
#!/bin/bash -l
module load gcc/16.2.0
module load python/3.14.7
source /scratch/hyt6ex/fibcon-main/venv/bin/activate
export OPENBLAS_NUM_THREADS=4
exec python -m ipykernel_launcher "$@"

2.	Run chmod +x ~/fibcon_kernel.sh and python -m ipykernel install --user --name fibcon --display-name "Fibcon (venv)".
3.	In ~/.local/share/jupyter/kernels/fibcon/kernel.json, set argv to ["/home/hyt6ex/fibcon_kernel.sh", "-f", "{connection_file}"].
4.	In the notebook, pick Fibcon (venv) at the top right.
RBT_LG_analysis-adata.ipynb from BAA_LAD_analysis-adata.ipynb. Unchanged
Area	Change	Why
Loading	Reads RBT/output/; the loading loop is uncommented	The loop was commented out, so adata_list was never defined
Save paths	RBT/analysis/	Old paths pointed to Michael's and AIM1C folders
utils import	sys.path.insert(0, '/scratch/hyt6ex/fibcon-main')	So from utils import ... finds Fibcon's file
Animal name	3rd part of the name (BP01)	The 1st part is Aim1c for every section
Keep largest section	Keyed by animal + ctrl/expt	Old key merged ctrl and expt of the same animal
Groups	group_names (BP, CG, NR) replaces baa_categories	BAA animal IDs don't apply
TA section	Removed	data is LG only
BAA13 / BAA25 drops	Removed	Specific to Michael's data
Hard-coded results	sig_var, posthoc_categories	Set from own ANOVA
LISA counts	.get(cat, 0)	Crashed when a section lacked a category
plot_LISA colors	Built from categories present	Colors shifted when a category was absent
minFD plot titles	adata_list[i]	Every plot showed section 2's title
Moran cells	Built with get_graph_W, Moran, Moran_Local	Called functions missing from utils.py
counts_df_gn, pingouin	Defined / imported in the LG section	Used but defined only in removed TA cells
The commented-out calculate_LISA(...) lines were dropped because LISA labels are already stored in each .h5ad (obs['LISA-minFD']). The num_cln line needs nuclei masks, which RBT does not have.

Errors w/ jupyter NB:

/home/hyt6ex/.local/lib/python3.13/site-packages/squidpy/pl/_spatial_utils.py:468: FutureWarning: The method obs_vector is deprecated and will be removed in the future. Use anndata.acc.A instead of obs_vector. E.g. `vec = adata[A.obs['foo']]` or `vec = adata[A.layers['l']['bar', :]]`
  color_source_vector = adata.obs_vector(value_to_plot, layer=layer)

** future warning don’t need to fix 
Terminal [KME(2.1] get the venv environment as an option for the kernel then select in the top-right drop-down: 
python -m ipykernel install --user --name fibcon --display-name "Fibcon (venv)"

-	Works in kernel – fix titles and match the maps 
# before
for i in specific_idx:
    sq.pl.spatial_scatter(adata_list[i], 
                          color='minFD', 
                          shape=None,
                          size=6,
                          size_key='tissue_hires_scalef',
                          library_key='section_fname',
                          title=next(iter(adata_list[2].uns['spatial'])),
                          #title='VML Injury',
                          cmap='RdBu_r',
                          connectivity_key='delaunay_connectivities',
                          edges_width=.25,
                          vmin=5,    
                          vmax=60,
                         )
    plt.show()


# after: uses index of sections to loop through and create new list – before wasn’t looping through the right
for i in specific_idx: 
    ad_i = adata_list[i] 
    section_name = ad_i.obs['section_fname'].iloc[0]
    print(i, section_name, f'{ad_i.n_obs} fibers')

    sq.pl.spatial_scatter(ad_i, 
                          color='minFD', 
                          shape=None,
                          size=6,
                          size_key='tissue_hires_scalef',
                          library_key='section_fname',
                          title=section_name, # new variable defined above 
                          #title='VML Injury',
                          cmap='RdBu_r',
                          connectivity_key='delaunay_connectivities',
                          edges_width=.25,
                          vmin=5,    
                          vmax=60,
                         )
    plt.show()




# issue with loading from utils.py to get moran/make graph ** missing fxns get_graph_moran and make_moran_scatterplot  also error loading lippysals

#before
import sys

from libpysal.weights import W, lag_spatial # was getting error so in the terminal  %pip install libpysal esda splot opencv-python-headless fastremap # bc not in venv envir in nb
from esda.moran import Moran, Moran_Local
from splot.esda import moran_scatterplot


from utils import get_graph_W, calculate_LISA, get_graph_moran, make_moran_scatterplot


# calculate_LISA(adata_list[2], 'num_cln', neighbors_key='delaunay_neighbors')
#calculate_LISA(adata_list[2], 'minFD', neighbors_key='delaunay_neighbors') # calculates for one section

make_moran_scatterplot(adata_list[2], 'minFD')

# after
import sys
sys.path.insert(0, '/scratch/hyt6ex/fibcon-main') # so `from utils import ...` finds Fibcon's utils.py


from libpysal.weights import W, lag_spatial
from esda.moran import Moran, Moran_Local
from splot.esda import moran_scatterplot


from utils import get_graph_W, calculate_LISA # get_graph_moran, make_moran_scatterplot are not in this utils.py


# LISA labels are already stored in each .h5ad by the batch job.
# Uncommenting the next line recomputes them for one section and overwrites that section's labels in memory.
# calculate_LISA(adata_list[2], 'num_cln', neighbors_key='delaunay_neighbors')   # needs nuclei masks; not in RBT data
# calculate_LISA(adata_list[2], 'minFD', neighbors_key='delaunay_neighbors')


# replaces make_moran_scatterplot: global Moran's I scatterplot for one section
ad_i = adata_list[2]
G_i = nx.Graph(ad_i.obsp[ad_i.uns['delaunay_neighbors']['connectivities_key']])
w_i = get_graph_W(G_i)
y_i = ad_i[:, ['minFD']].to_df().values.flatten()

moran_i = Moran(y_i, w_i)
fig, ax = moran_scatterplot(moran_i)
ax.set_title(f"{ad_i.obs['section_fname'].iloc[0]}  Moran's I = {moran_i.I:.2f}")
plt.show()
# this replaces the make_moran_scatterplot

# Local Moran (LISA) scatterplot (replaces get_graph_moran + visualize_moran_scatter)
moran_loc_i = Moran_Local(y_i, w_i)
fig, ax = moran_scatterplot(moran_loc_i, p=0.05)
ax.set_title(ad_i.obs['section_fname'].iloc[0])
plt.show()

#one number for the whole section: do fibers tend to sit next to fibers of similar size?

NEXT CELL 

# import error, yellow = commented out but was in original cell 

# from utils import visualize_moran_scatter          # not in this utils.py
#
# adata_moran = get_graph_moran(adata_list[1], feature='minFD')
# visualize_moran_scatter(adata_moran)
# plt.show()

# Local Moran (LISA) scatterplot (replaces get_graph_moran + visualize_moran_scatter)
moran_loc_i = Moran_Local(y_i, w_i)
fig, ax = moran_scatterplot(moran_loc_i, p=0.05)
ax.set_title(ad_i.obs['section_fname'].iloc[0])
plt.show()


# adata_moran = get_graph_moran(adata_list[44], feature='minFD')
# visualize_moran_scatter(adata_moran)
# plt.show()


-	Group all ctrls 
-	EXPT groups: (5)
-	12 week 
-	BP 
-	CG
-	NR
-	All 20 week 
-	CG
-	NR

 # before
ad_keep_dict = {}


for ad_i in adata_list:
    ad_full_name = ad_i.obs['section_fname'].iloc[0]
    ad_name_parts = ad_full_name.split("_")
    ad_name = '_'.join(ad_name_parts[:3])
    
    (ad_keep_value, _) = ad_keep_dict.get(ad_name, (0, ''))
    if ad_i.n_obs > ad_keep_value:
        ad_keep_dict[ad_name] = (ad_i.n_obs, ad_i.obs['section_fname'].iloc[0])
        
ad_keep_dict


print(f'Number of images after: {len(adata_list)}')
print(f'Number of images after: {len(ad_keep_dict)}')


# after - yellow = changed, before didn’t account for condition now output = all
# 65 rather than 37 before


# Keep one section per animal per condition (the one with the most fibers).
# Key = animal (3rd part of the name) + ctrl/expt (4th part), so ctrl and expt are kept separately.
ad_keep_dict = {}


for ad_i in adata_list:
    ad_full_name = ad_i.obs['section_fname'].iloc[0]
    ad_name_parts = ad_full_name.split("_")
    condition = 'ctrl' if 'ctrl' in ad_name_parts[3].lower() else 'expt'
    ad_name = ad_name_parts[2] + '_' + condition


    (ad_keep_value, _) = ad_keep_dict.get(ad_name, (0, ''))
    if ad_i.n_obs > ad_keep_value:
        ad_keep_dict[ad_name] = (ad_i.n_obs, ad_full_name) # keeps image with more fibers if there are two of the same


print(f'Number of images before: {len(adata_list)}')
print(f'Number of images after: {len(ad_keep_dict)}')




# next cell 


# before - wrong directory path and directory didn’t exist 


import pickle


# Save list to pickle 
list_pkl_file = '/scratch/hyt6ex/AIM1C_laminin_all/ad_shortlist.pkl'
with open(list_pkl_file, 'wb') as f:
    pickle.dump(adata_short_list, f)






# after - 
​​import pickle


ANALYSIS_DIR = Path("/scratch/hyt6ex/RBT/analysis/")
ANALYSIS_DIR.mkdir(exist_ok=True)


# Save list to pickle
list_pkl_file = ANALYSIS_DIR / 'ad_shortlist.pkl'
with open(list_pkl_file, 'wb') as f:
    pickle.dump(adata_short_list, f) # Serializes an object and writes it directly to an open binary file




# SORTING CATEGORIES: 
# sorting groups 
# 1: ctrl (all ctrl groups)
# 2: BP (all BP, all 12 wks)
# 3: CG 12wk (2, 13-17, 20-21)
# 4: NR 12 wk (1-7)
# 5: CG 20 wk (5-12)
# 6: NR 20 wk (9-16)


import re 


cg_12wk = [2,13,14,15,16,17,20,21]
nr_12wk = list(range(1,8))
cg_20wk = list(range(5,13))
nr_20wk = list(range(9,17))


group_order = ['Control', 'BP', 'CG 12wk', 'NR 12wk', 'CG 20wk', 'NR 20wk']


def get_animal(section_fname):
    # Aim1c_LG_BP01_ctrl_s1c1-3 -> BP01
    return section_fname.split('_')[2]


def get_group(animal_name):
    # BP01 -> BP -> group_names['BP']
    prefix = re.match(r'[A-Za-z]+', animal_name).group()
    num = int(re.search(r'\d+', animal_name).group())
    if prefix == 'BP': 
        return 'BP'
    if prefix == 'CG' and num in cg_12wk: 
        return 'CG 12wk'
    if prefix == 'NR' and num in nr_12wk: 
        return 'NR 12wk'
    if prefix == 'CG' and num in cg_20wk: 
        return 'CG 20wk'
    if prefix == 'NR' and num in nr_20wk: 
        return 'NR 20wk'
    if prefix == 'BP': 
        return 'BP'
    return 'NO GROUP'


 def sort_condition(section_fname):
    if 'ctrl' in section_fname.lower():
        return 'CTRL'
    if 'exp' in section_fname.lower():
        return 'EXP'


def section_group(section_fname):
    if sort_condition(section_fname) == 'CTRL':
        return 'Control'
    return get_group(get_animal(section_fname))




# Before: (added cell above to sort ^)
adatas = ad.concat(adata_short_list, merge='same', uns_merge='first')
adatas.obs['section_fname'] = adatas.obs['section_fname'].astype('category')
adatas.obs_names_make_unique()


# def sort_condition(section_fname):
#     if 'ctrl' in section_fname.lower():
#         return 'CTRL'
#     if '_l_' in section_fname.lower():
#         return 'CTRL'
#     if 'expt' in section_fname.lower():
#         return 'EXP'
#     if '_t_' in section_fname.lower():
#         return 'EXP'


def get_animal(section_fname):
    # Aim1c_LG_BP01_ctrl_s1c1-3 -> BP01
    return section_fname.split('_')[2]


def get_group(animal_name):
    # BP01 -> BP -> group_names['BP']
    prefix = re.match(r'[A-Za-z]+', animal_name).group()
    return group_names.get(prefix, prefix)


def sort_condition(section_fname):
    if 'ctrl' in section_fname.lower():
        return 'CTRL'
    if 'exp' in section_fname.lower():
        return 'EXP'
    
adatas_df = adatas.to_df()
adatas_df['section_fname'] = adatas.obs['section_fname']
adatas_df['condition'] = adatas_df['section_fname'].map(lambda x: sort_condition(x))
adatas_df['muscle'] = adatas_df['section_fname'].map(lambda x: 'TA' if 'ta' in x.lower() else 'LGN')
adatas_df['animal_name'] = adatas_df['section_fname'].map(lambda x: x.split('_')[0])
adatas_df['group'] = adatas_df['animal_name'].map(baa_categories)
adatas_df['LISA-minFD'] = adatas.obs['LISA-minFD']
adatas_df.head()


adatas.write_h5ad('/scratch/hyt6ex/AIM1C_laminin_all/all_sections_adata.h5ad')


# changes to above cell: 


# adatas = ad.concat(adata_short_list, merge='same', uns_merge='first')
# adatas.obs['section_fname'] = adatas.obs['section_fname'].astype('category')
# adatas.obs_names_make_unique() 
# ^ was getting error 
adatas = ad.concat(adata_short_list, merge='same', uns_merge='first', index_unique='-')
adatas.obs['section_fname'] = adatas.obs['section_fname'].astype('category')
    
adatas_df = adatas.to_df()
adatas_df['section_fname'] = adatas.obs['section_fname']
adatas_df['condition'] = adatas_df['section_fname'].map(lambda x: sort_condition(x))
adatas_df['muscle'] = adatas_df['section_fname'].map(lambda x: 'TA' if 'ta' in x.lower() else 'LGN')
adatas_df['animal_name'] = adatas_df['section_fname'].map(lambda x: get_animal(x))
adatas_df['group'] = adatas_df['section_fname'].map(section_group)
adatas_df['LISA-minFD'] = adatas.obs['LISA-minFD']


adatas.write_h5ad(ANALYSIS_DIR / 'all_sections_adata.h5ad')
adatas_df.head()
adatas_df.groupby('group')['animal_name'].count()


print(adatas_df.groupby('group')['animal_name'].unique())








 

START HERE FOR LOADING DATA FROM FILE: 



# rerun setup cell, deleted Michael’s baa_categories (our groups defined above and # again below)
import re
from pathlib import Path
import pickle


import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squidpy as sq
import pingouin as pg
from scipy import stats
from tqdm import tqdm
from statannotations.Annotator import Annotator


ANALYSIS_DIR = Path("/scratch/hyt6ex/RBT/analysis/")


#before 
section_data_dicts = []
for adata_i in tqdm(adata_list):
    slide_name = adata_i.obs['section_fname'].iloc[0]
    is_control = "ctrl" in slide_name.lower()
    
    lisa_counts = adata_i.obs['LISA-minFD'].value_counts()
    
    this_section_data_dict = {
        'condition' : 'CTRL' if is_control else 'EXP',
        'slide_name': slide_name,
        'ns_count' : lisa_counts['ns'],
        'HH_count' : lisa_counts['HH'],
        'LL_count' : lisa_counts['LL'],
        'LH_count' : lisa_counts['LH'],
        'HL_count' : lisa_counts['HL'],
        'animal_name' : get_animal(slide_name),
    }
    section_data_dicts.append(this_section_data_dict)
    
counts_df = pd.DataFrame(section_data_dicts)
counts_df['Total Count'] = counts_df['ns_count'] + counts_df['HH_count'] + counts_df['LL_count'] + counts_df['LH_count'] + counts_df['HL_count']
counts_df['ns_fraction'] = counts_df['ns_count'] / counts_df['Total Count']
counts_df['HH_fraction'] = counts_df['HH_count'] / counts_df['Total Count']
counts_df['LL_fraction'] = counts_df['LL_count'] / counts_df['Total Count']
counts_df['LH_fraction'] = counts_df['LH_count'] / counts_df['Total Count']
counts_df['HL_fraction'] = counts_df['HL_count'] / counts_df['Total Count']


counts_df['group'] = counts_df['animal_name'].map(baa_categories)
counts_df['muscle'] = counts_df['group'].map(lambda x: 'TA' if 'ta' in x.lower() else 'LGN')


counts_df.sort_values(['animal_name', 'group', 'condition'])


counts_df['group'] = counts_df.apply(lambda x: 'Control' if x['condition']=='CTRL' else x['group'], axis=1)




counts_df.head()


#after 


section_data_dicts = []
for adata_i in tqdm(adata_list):
    slide_name = adata_i.obs['section_fname'].iloc[0]
    lisa_counts = adata_i.obs['LISA-minFD'].value_counts()


    this_section_data_dict = {
        'condition'   : sort_condition(slide_name),
        'slide_name'  : slide_name,
        'ns_count'    : lisa_counts.get('ns', 0),
        'HH_count'    : lisa_counts.get('HH', 0),
        'LL_count'    : lisa_counts.get('LL', 0),
        'LH_count'    : lisa_counts.get('LH', 0),
        'HL_count'    : lisa_counts.get('HL', 0),
        'animal_name' : get_animal(slide_name),
        'group'       : section_group(slide_name),
    }
    section_data_dicts.append(this_section_data_dict)


counts_df = pd.DataFrame(section_data_dicts)
counts_df['Total Count'] = counts_df['ns_count'] + counts_df['HH_count'] + counts_df['LL_count'] + counts_df['LH_count'] + counts_df['HL_count']
counts_df['ns_fraction'] = counts_df['ns_count'] / counts_df['Total Count']
counts_df['HH_fraction'] = counts_df['HH_count'] / counts_df['Total Count']
counts_df['LL_fraction'] = counts_df['LL_count'] / counts_df['Total Count']
counts_df['LH_fraction'] = counts_df['LH_count'] / counts_df['Total Count']
counts_df['HL_fraction'] = counts_df['HL_count'] / counts_df['Total Count']
counts_df['muscle'] = 'LGN'


counts_df.head()


# changes made 
from tqdm import tqdm 


section_data_dicts = []
for adata_i in tqdm(adata_list):
    slide_name = adata_i.obs['section_fname'].iloc[0]
    is_control = "ctrl" in slide_name.lower()
    
    lisa_counts = adata_i.obs['LISA-minFD'].value_counts()
    
    this_section_data_dict = {
        'condition' : sort_condition(slide_name), # changed variable 
        'slide_name': slide_name,
        'ns_count' : lisa_counts['ns'],
        'HH_count' : lisa_counts['HH'],
        'LL_count' : lisa_counts['LL'],
        'LH_count' : lisa_counts['LH'],
        'HL_count' : lisa_counts['HL'],
        'animal_name' : get_animal(slide_name),
        'group' : section_group(slide_name)
    }
    section_data_dicts.append(this_section_data_dict)
    
counts_df = pd.DataFrame(section_data_dicts)
counts_df['Total Count'] = counts_df['ns_count'] + counts_df['HH_count'] + counts_df['LL_count'] + counts_df['LH_count'] + counts_df['HL_count']
counts_df['ns_fraction'] = counts_df['ns_count'] / counts_df['Total Count']
counts_df['HH_fraction'] = counts_df['HH_count'] / counts_df['Total Count']
counts_df['LL_fraction'] = counts_df['LL_count'] / counts_df['Total Count']
counts_df['LH_fraction'] = counts_df['LH_count'] / counts_df['Total Count']
counts_df['HL_fraction'] = counts_df['HL_count'] / counts_df['Total Count']


# counts_df['group'] = counts_df['animal_name'].section_group(slide_name)
# counts_df['muscle'] = counts_df['group'].map(lambda x: 'TA' if 'ta' in x.lower() else 'LGN')


# counts_df.sort_values(['animal_name', 'group', 'condition'])


counts_df['muscle'] = 'LGN'
counts_df.head()


print(counts_df.head()) # holds spatial pattern data before statistics are run


STATISTICAL ANALYSIS: old had both ta and lgn analysis, adapted/kept a couple cells (shown below) and deleted the rest of the ta cells → adjust/adapt the lgn section for new data

# from ta analysis - keep and edit 
import squidpy as sq
from scipy import stats




cluster_key = 'LISA-minFD'
    
dict_list = []
for ad_i in tqdm(adata_list):
    features_dict = {}
    


    
    file_name = ad_i.obs['section_fname'].unique()[0]
    animal_name = file_name.split('_')[0]
    
    if 'BAA13' == animal_name and 'ctrl' in file_name.lower():
        print('Skipping BAA13 control')
        continue
    elif 'BAA25' == animal_name and 'exp' in file_name.lower():
        print('Skipping BAA25 exp')
        continue


    sq.gr.centrality_scores(ad_i, 
                        cluster_key=cluster_key, 
                        connectivity_key='delaunay'
    )
    ad_i.uns[cluster_key + '_centrality_scores']['cluster'] = ad_i.uns[cluster_key + '_centrality_scores'].index


    group = baa_categories[animal_name]
    if 'ctrl' in file_name:
        group = 'Control' 
        
    features_dict['animal_name'] = animal_name
    features_dict['muscle'] = 'TA' if 'ta' in file_name.lower() else 'LGN'
    features_dict['group'] = group


    ad_i_df = ad_i.to_df()
    ad_i_df[cluster_key] = ad_i.obs[cluster_key]


    for c in ad_i_df[cluster_key].unique():
        cluster_minfd_distrib = ad_i_df[ad_i_df[cluster_key] == c]['minFD']
        
        kurtosis = stats.kurtosis(cluster_minfd_distrib)
        skew = stats.skew(cluster_minfd_distrib)
        mean = cluster_minfd_distrib.mean()
        median = np.median(cluster_minfd_distrib)
        mode = stats.mode(cluster_minfd_distrib).mode
        variance = cluster_minfd_distrib.var()
        fraction = len(ad_i_df[ad_i_df[cluster_key] == c]) / len(ad_i_df)


        features_dict[cluster_key + '-cluster' + str(c) + '_kurtosis'] = kurtosis
        features_dict[cluster_key + '-cluster' + str(c) + '_skew'] = skew
        features_dict[cluster_key + '-cluster' + str(c) + '_mean'] = mean
        features_dict[cluster_key + '-cluster' + str(c) + '_median'] = median
        features_dict[cluster_key + '-cluster' + str(c) + '_mode'] = mode
        features_dict[cluster_key + '-cluster' + str(c) + '_variance'] = variance
        features_dict[cluster_key + '-cluster' + str(c) + '_fraction'] = fraction
        
        features_dict[cluster_key + '-cluster' + str(c) + '_distance_median'] = np.median(ad_i.uns['lisa_distance_dict'][c])
        features_dict[cluster_key + '-cluster' + str(c) + '_distance_variance'] = np.var(ad_i.uns['lisa_distance_dict'][c])
        
        centrality_scores_cluster = ad_i.uns[cluster_key + '_centrality_scores'][ad_i.uns[cluster_key + '_centrality_scores']['cluster'] == c]
        features_dict[cluster_key + '-cluster' + str(c) + '_degree_centrality'] = centrality_scores_cluster['degree_centrality'].item()
        features_dict[cluster_key + '-cluster' + str(c) + '_average_clustering'] = centrality_scores_cluster['average_clustering'].item()
        features_dict[cluster_key + '-cluster' + str(c) + '_closeness_centrality'] = centrality_scores_cluster['closeness_centrality'].item()


    features_dict['total_FCSA'] = ad_i_df['area'].sum()
        
    dict_list.append(features_dict)
    


lisa_features_df = pd.DataFrame(dict_list)
print(lisa_features_df.columns)
lisa_features_df.head()




# from ta analysis - keep and edit 
closeness_feats = [f'LISA-minFD-cluster{c}_closeness_centrality' for c in ['ns', 'LL', 'LH', 'HH', 'HL']] 
melted_df = lisa_features_df.melt(
    id_vars=['animal_name', 'muscle', 'group'],
    value_vars=closeness_feats,
    var_name='Category',
    value_name='Closeness Centrality'
)


# Rename categories for better readability
category_mapping = {
    'LISA-minFD-clusterns_closeness_centrality': 'ns',
    'LISA-minFD-clusterLL_closeness_centrality': 'LL',
    'LISA-minFD-clusterLH_closeness_centrality': 'LH',
    'LISA-minFD-clusterHH_closeness_centrality': 'HH',
    'LISA-minFD-clusterHL_closeness_centrality': 'HL',
}
melted_df['Category'] = melted_df['Category'].map(category_mapping)


melted_df_ta = melted_df[melted_df['muscle'] == 'TA']
melted_df_gn = melted_df[melted_df['muscle'] == 'LGN']

