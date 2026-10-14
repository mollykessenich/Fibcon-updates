from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
from tqdm.auto import tqdm
from skimage.measure import regionprops_table, intersection_coeff, label, regionprops, find_contours
from tqdm import tqdm
from pathlib import Path
from skimage.color import rgb2gray 
import numpy as np
import cv2 
import fastremap

from scipy.spatial import KDTree, Delaunay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.cluster import AgglomerativeClustering, KMeans
from tqdm.contrib.concurrent import process_map  

from scipy.spatial.qhull import QhullError

import numpy as np

import warnings
import functools

from libpysal.weights import W, lag_spatial
from esda.moran import Moran, Moran_Local


# def deprecated(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         warnings.warn(f"{func.__name__} is deprecated and should not be used.", DeprecationWarning, stacklevel=2)
#         return func(*args, **kwargs)
#     return wrapper


# Function definitions: 
def min_FD(image):
    """find the minimum feret diameter from a single object in an image"""
    contours, _ = cv2.findContours(image.astype(np.uint8), 
                                   mode=cv2.RETR_EXTERNAL, 
                                   method=cv2.CHAIN_APPROX_NONE)

    # Find the minimum enclosing rectangle of the object
    rect = cv2.minAreaRect(contours[0])

    # Compute the coordinates of the corners of the rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Compute the pairwise distances between the corners of the rectangle
    distances = []
    for i in range(4):
        for j in range(i+1, 4):
            distances.append(np.linalg.norm(box[i] - box[j]))

    return min(distances)


def get_minFD(points):
    """
    Computes the minimum Feret diameter of a list of pixels representing a segmented object.
    
    Parameters:
        points (ndarray): A list of 2D points representing the pixels of an object.
        
    Returns:
        The minimum Feret diameter of the object.
    """
    # Convert the list of points to a numpy array
    points = np.asarray(points)
    
    # Find the minimum enclosing rectangle of the points
    rect = cv2.minAreaRect(points)
    
    # Compute the coordinates of the corners of the rectangle
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Compute the pairwise distances between the corners of the rectangle
    distances = []
    for i in range(4):
        for j in range(i+1, 4):
            distances.append(np.linalg.norm(box[i] - box[j]))
    
    return min(distances)


def get_dilation(coords, dilation=1):
    """Coords should be 2D np array in (row, col) format
    will return pixels around the original coords"""
    from scipy.ndimage import binary_dilation
    import math
    
    min_x = min(coords[:, 0])
    max_x = max(coords[:, 0])
    min_y = min(coords[:, 1])
    max_y = max(coords[:, 1])

    x_size = max_x - min_x
    y_size = max_y - min_y
    mask = np.zeros((x_size+dilation*2, y_size+dilation*2), dtype=int) # TODO: how to determine if dilation is enough?
    adj_coords = np.subtract(coords, [min_x-dilation, min_y-dilation])
    col_indicies = adj_coords[:,1]
    row_indicies = adj_coords[:,0]
    mask[row_indicies, col_indicies] = 1
    dil = binary_dilation(mask, iterations=dilation)
    dil = dil - mask
    dil_indicies = np.asarray(np.where(dil)).T
    dil_out = np.add(dil_indicies, [min_x-dilation, min_y-dilation])
    return dil_out


def get_edge(coords):
    """Coords should be 2D np array in (row, col) format
    will return pixels around the original coords"""
    from scipy.ndimage import binary_erosion
    import math
    
    min_x = min(coords[:, 0])
    max_x = max(coords[:, 0])
    min_y = min(coords[:, 1])
    max_y = max(coords[:, 1])

    x_size = max_x - min_x
    y_size = max_y - min_y
    mask = np.zeros((x_size+1, y_size+1), dtype=int) 
    adj_coords = np.subtract(coords, [min_x, min_y])
    col_indicies = adj_coords[:,1]
    row_indicies = adj_coords[:,0]
    mask[row_indicies, col_indicies] = 1
    ero = binary_erosion(mask)
    mask_out = mask - ero
    mask_indicies = np.asarray(np.where(mask_out)).T
    mask_out = np.add(mask_indicies, [min_x, min_y])
    return mask_out


def get_min(coords, column):
    assert column == 0 or column == 1
    return coords[:, column].min()


def get_max(coords, column):
    assert column == 0 or column == 1
    return coords[:, column].max()


def intersect2D(a, b):
    """
    Find row intersection between 2D numpy arrays, a and b.
    Returns another numpy array with shared rows
    
    Borrowed from https://gist.github.com/Robaina/b742f44f489a07cd26b49222f6063ef7
    """
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])


def are_rows_intersecting(edge_row, dilated_row):
    """Rows are pandas series""" 
    x_overlaps = dilated_row['min_x'] <= edge_row['max_x'] and dilated_row['max_x'] >= edge_row['min_x']
    y_overlaps = dilated_row['min_y'] <= edge_row['max_y'] and dilated_row['max_y'] >= edge_row['min_y']
    
    if not (x_overlaps and y_overlaps):
        return False
    
    # This way there won't be self-matches:
    touching = intersect2D(edge_row['edge'], dilated_row['dilated_coords']).size > 0
    return touching 


def are_coords_touching(edge_coords, dilated_coords):
    """coords should be numpy (length)-by-2 (rows, cols)"""
    x_overlaps = min(dilated_coords[:, 0]) <= max(edge_coords[:, 0]) \
        and max(dilated_coords[:, 0]) >= min(edge_coords[:, 0])
    y_overlaps = min(dilated_coords[:, 1]) <= max(edge_coords[:, 1]) \
        and max(dilated_coords[:, 1]) >= min(edge_coords[:, 1])
    
    if not (x_overlaps and y_overlaps):
        return False
    
    # This way there won't be self-matches:
    touching = intersect2D(edge_row['edge'], dilated_row['dilated_coords']).size > 0
    return touching 


def make_touching_dict(edge_indicies, np_edge, np_nearby_centroids, np_dilated_coords):
    touching_dict = {}
    for this_row, edge_row_coords in zip(edge_indicies, np_edge):

        nearby_this_indicies = np_nearby_centroids[this_row]
        possible_match_coords_list = np_dilated_coords[nearby_this_indicies] # The actual coords list, one per item

        touching_list = []
        for this_index in nearby_this_indicies:
            nearby_coords = np_dilated_coords[this_index]
            touching = intersect2D(edge_row_coords, nearby_coords).size > 0
            if touching:
                touching_list.append(this_index)
        touching_dict[this_row] = touching_list
    return touching_dict


def associate_fiber_nuclei(fiber_masks, nuclei_masks, periphery_threshold=0.55, center_threshold=0.55):
    """ iterate over each fiber and find nuclei that lie on the perimeter of the fiber 
    or are fully inside the median nuclei equivalent diameter

    Parameters
    ----------
    fiber_masks : 2D array_like
        Fiber mask input array.
    nuclei_masks : 2D array_like
        Nuclei mask input array.
    periphery_threshold : float, optional
        Pixel fraction of the nucleus that must in a fiber's periphery to be included. 
    center_threshold : float, optional
        Pixel fraction of the nucleus that must in a fiber's core to be included. 

    Returns
    -------
    fiber_center_nuceli : dict[int, List[int]]
        fiber_id:nuclei_ids for center nuclei
    fiber_ring_nuclei : dict[int, List[int]]
        fiber_id:nuclei_ids for peripherial nuclei
    """

    # validate input:
    try:
        periphery_in_range = 0. < periphery_threshold < 1.0
        center_in_range = 0. < center_threshold < 1.0
        if not periphery_in_range:
            raise ValueError(f"periphery_threshold must be a float in the interval (0, 1) but was {periphery_threshold}")
        if not center_in_range:
            raise ValueError(f"center_threshold must be a float in the interval (0, 1) but was {center_threshold}")

        input_same_shape =  fiber_masks.shape == nuclei_masks.shape
        if not input_same_shape:
            raise ValueError(f"fiber_masks and nuclei_masks must be the same shape but were {fiber_masks.shape} "
                             f"and {nuclei_masks.shape}")

    except ValueError as e:
        raise e

    #find median area of nuclei
    n_ids, nuc_areas = np.unique(nuclei_masks, return_counts=True)
    n_ids, nuc_areas = n_ids[1:], nuc_areas[1:]
    median_nuc_area = np.median(nuc_areas)
    
    d_eq = int(2*np.sqrt(median_nuc_area/np.pi))
    
    fiber_idxs = np.unique(fiber_masks)
    fiber_idxs = fiber_idxs[1:]
    eroded_fiber_masks = np.zeros_like(fiber_masks)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (d_eq, d_eq))
    fiber_center_nuceli = {}
    fiber_ring_nuclei = {}

    for idx in tqdm(fiber_idxs, 
                    ascii=True, 
                    position=0, 
                    leave=True,
                    desc="Finding associated nuclei for all fibers"):

        label_mask = (fiber_masks == idx).astype(np.uint8)
        eroded_mask = cv2.erode(label_mask, kernel, iterations=1)
        eroded_fiber_masks[eroded_mask == True] = idx
        img_h, img_w = fiber_masks.shape
    
        # get bounding box of eroded mask
        x, y, w, h = cv2.boundingRect(label_mask)
    
        # get nuclei that are within the bounding box
        # make sure to only get nuclei that are fully within the fiber
        # expand the bounding box by d_eq pixels to make sure we get all the nuclei
        # catch the cases where the bounding box is at the edge of the image
        y_low = y - 2*d_eq
        if y_low < 0:
            y_low = 0
        y_high = y + h + 2*d_eq
        if y_high > img_h:
            y_high = img_h
        x_low = x - 2*d_eq
        if x_low < 0:
            x_low = 0
        x_high = x + w + 2*d_eq
        if x_high < img_w:
            x_high = img_w
    
        nuclei_bbox = nuclei_masks[y_low:y_high, x_low:x_high]
        mask_bbox = label_mask[y_low:y_high, x_low:x_high]
        eroded_mask_bbox = eroded_mask[y_low:y_high, x_low:x_high]
        ring_bbox = mask_bbox - eroded_mask_bbox
    
        # find nuclei that are fully within the fiber
        nuclei_in_bbox = np.unique(nuclei_bbox)
        try:
            nuclei_in_bbox = nuclei_in_bbox[1:]
        except IndexError:
            nuclei_in_bbox = []
            continue
    
        centrally_located_nuclei = []
        ring_nuclei = []
        for nuclei in nuclei_in_bbox:
            nuclei_mask = (nuclei_bbox == nuclei).astype(bool)

            try:
                ring_intersection = intersection_coeff(nuclei_mask, ring_bbox)
                # ring_intersection = np.sum(nuclei_mask & ring_bbox) / np.sum(nuclei_mask)
                center_intersection = intersection_coeff(nuclei_mask, eroded_mask_bbox)

            except ValueError:
                print(f"Skipping nuclei overlap calculation nuclei={nuclei}, fiber idx={idx}, "
                      f"eroded_mask_bbox={eroded_mask_bbox.shape}, "
                      f"nuclei_mask={nuclei_mask.shape}, ring_bbox={ring_bbox.shape}")
                continue

            if center_intersection >= center_threshold:
                centrally_located_nuclei.append(nuclei)
                continue
            if ring_intersection + center_intersection >= periphery_threshold:
                ring_nuclei.append(nuclei)
    
    
        fiber_center_nuceli[idx] = centrally_located_nuclei
        fiber_ring_nuclei[idx] = ring_nuclei

    return fiber_center_nuceli, fiber_ring_nuclei


def process_fiber(args):
    fiber_id, fiber_masks, nuclei_masks, d_eq, periphery_threshold, center_threshold = args

    
    
    label_mask = (fiber_masks == fiber_id).astype(np.uint8)
    eroded_mask = cv2.erode(label_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (d_eq, d_eq)), iterations=1)
    img_h, img_w = fiber_masks.shape

    # Get bounding box of eroded mask
    x, y, w, h = cv2.boundingRect(label_mask)
    # Calculate expanded bounding box coordinates
    y_low, y_high = max(0, y - 2*d_eq), min(img_h, y + h + 2*d_eq)
    x_low, x_high = max(0, x - 2*d_eq), min(img_w, x + w + 2*d_eq)

    nuclei_bbox = nuclei_masks[y_low:y_high, x_low:x_high]
    mask_bbox = label_mask[y_low:y_high, x_low:x_high]
    eroded_mask_bbox = eroded_mask[y_low:y_high, x_low:x_high]
    ring_bbox = mask_bbox - eroded_mask_bbox

    centrally_located_nuclei = []
    ring_nuclei = []

    nuclei_in_bbox = np.unique(nuclei_bbox)
    nuclei_in_bbox = nuclei_in_bbox[1:] if nuclei_in_bbox.size > 1 else []

    for nuclei in nuclei_in_bbox:
        nuclei_mask = (nuclei_bbox == nuclei).astype(bool)
        ring_intersection = intersection_coeff(nuclei_mask, ring_bbox)
        center_intersection = intersection_coeff(nuclei_mask, eroded_mask_bbox)

        if center_intersection >= center_threshold:
            centrally_located_nuclei.append(nuclei)
            continue
        if ring_intersection + center_intersection >= periphery_threshold:
            ring_nuclei.append(nuclei)

    return fiber_id, centrally_located_nuclei, ring_nuclei


def associate_fiber_nuclei_parallel(fiber_masks, nuclei_masks, 
                                    periphery_threshold=0.55, center_threshold=0.55, 
                                    chunksize=100, threads=4):
    """ 
    TODO: description

    Parameters
    ----------
    fiber_masks : 2D array_like
        Fiber mask input array.
    nuclei_masks : 2D array_like
        Nuclei mask input array.
    periphery_threshold : float, optional
        Pixel fraction of the nucleus that must in a fiber's periphery to be included. 
    center_threshold : float, optional
        Pixel fraction of the nucleus that must in a fiber's core to be included. 
    chunksize : int, optional
        The number of fibers to process in a chunk. Lower the value if memory errors are encountered. 
    threads : int, optional
        The number of threads to use in parallel processing. 

    Returns
    -------
    fiber_center_nuceli : dict[int, List[int]]
        fiber_id:nuclei_ids for center nuclei
    fiber_ring_nuclei : dict[int, List[int]]
        fiber_id:nuclei_ids for peripherial nuclei
    """

    # validate input:
    try:
        periphery_in_range = 0. < periphery_threshold < 1.0
        center_in_range = 0. < center_threshold < 1.0
        if not periphery_in_range:
            raise ValueError(f"periphery_threshold must be a float in the interval (0, 1) but was {periphery_threshold}")
        if not center_in_range:
            raise ValueError(f"center_threshold must be a float in the interval (0, 1) but was {center_threshold}")

        input_same_shape =  fiber_masks.shape == nuclei_masks.shape
        if not input_same_shape:
            raise ValueError(f"fiber_masks and nuclei_masks must be the same shape but were {fiber_masks.shape} "
                             f"and {nuclei_masks.shape}")

    except ValueError as e:
        raise e

    #find median area of nuclei
    n_ids, nuc_areas = np.unique(nuclei_masks, return_counts=True)
    n_ids, nuc_areas = n_ids[1:], nuc_areas[1:]
    median_nuc_area = np.median(nuc_areas)
    
    d_eq = int(2*np.sqrt(median_nuc_area/np.pi))
    
    fiber_idxs = np.unique(fiber_masks)[1:]  # Excluding the background
    
    args = [(idx, fiber_masks, nuclei_masks, d_eq, periphery_threshold, center_threshold) for idx in fiber_idxs]

    # Use process_map for parallel execution with a progress bar
    results = process_map(process_fiber, args, 
                          max_workers=threads, chunksize=chunksize, 
                          desc="finding nuclei in fibers")

    fiber_center_nuceli = {}
    fiber_ring_nuclei = {}

    for idx, center_nuclei, ring_nuclei in results:
        fiber_center_nuceli[idx] = center_nuclei
        fiber_ring_nuclei[idx] = ring_nuclei

    return fiber_center_nuceli, fiber_ring_nuclei


def calculate_angle(v1, v2):
    """Calculate the angle in degrees between vectors 'v1' and 'v2'"""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def make_graph(positions, keep_edges=6, distance_threshold=.15, angle_threshold=None):
    """
    Create a graph based on positions using Delaunay triangulation and then refine it
    by keeping only the 'keep_edges' shortest edges for each node and removing edges that
    exceed a distance threshold.
    
    Parameters:
        positions (array-like): The spatial positions of the nodes.
        keep_edges (int): The number of shortest edges to keep for each vertex. Default is 6.
        distance_threshold (float): The threshold ratio to the pseudo-diameter for removing edges. Default is 0.15.
        angle_threshold (float): The angle in degrees below which to remove edges. Default is None (no angle filtering).
        
    Returns:
        G (networkx.Graph): The refined graph.
    """
    # Create the Delaunay triangulation
    tri = Delaunay(positions)
    simplices = tri.simplices
    
    # Create a graph from the Delaunay triangulation
    G = nx.Graph()
    for simplex in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                p1, p2 = positions[simplex[i]], positions[simplex[j]]
                distance = np.linalg.norm(p1 - p2)
                G.add_edge(simplex[i], simplex[j], weight=distance)
                
    # Refine the graph based on keep_edges and distance_threshold
    for node in list(G.nodes):
        edges = list(G.edges(node, data=True))
        edges.sort(key=lambda x: x[2]['weight'])
        if len(edges) > keep_edges:
            for edge in edges[keep_edges:]:
                if edge[2]['weight'] > distance_threshold:
                    G.remove_edge(edge[0], edge[1])
                    
        # Angle-based edge removal (optional)
        if angle_threshold is not None:
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 1:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        v1 = positions[neighbors[i]] - positions[node]
                        v2 = positions[neighbors[j]] - positions[node]
                        angle = calculate_angle(v1, v2)
                        if angle < angle_threshold:
                            G.remove_edge(node, neighbors[i])
                            G.remove_edge(node, neighbors[j])

    return G
   

def find_density_core(graph):
    """
    Finds core section using density distribution of the given graph using Kernel Density Estimation (KDE).
    
    :param graph: A graph containing vertex positions and fiber sizes (minFD).
    :return: x_max_density, y_max_density - X and Y coordinates of the maximum density.
    """
    pos = np.array(list(graph.vp.v_pos))
    size = np.array(list(graph.vp.v_minFD))
    data = np.column_stack((pos, size[:, np.newaxis]))

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Create a grid of bandwidths to search
    params = {'bandwidth': np.logspace(-2, 2, 10),
              'kernel': ['gaussian', 'exponential', 'tophat']
             }
    grid_search = GridSearchCV(KernelDensity(), params)
    grid_search.fit(data_scaled)

    print(f"KDE best model: {grid_search.best_estimator_.kernel=}, {grid_search.best_estimator_.bandwidth=}")

    # Create a Kernel Density Estimator with the best bandwidth
    kde = KernelDensity(kernel=grid_search.best_estimator_.kernel, bandwidth=grid_search.best_estimator_.bandwidth)
    kde.fit(data_scaled)

    # Create a denser grid to evaluate the KDE
    x_grid, y_grid = np.meshgrid(np.linspace(pos[:, 0].min(), pos[:, 0].max(), 100),
                                 np.linspace(pos[:, 1].min(), pos[:, 1].max(), 100))
    xy_grid = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    xy_grid_scaled = scaler.transform(np.column_stack([xy_grid, np.zeros(xy_grid.shape[0])]))

    # Evaluate the KDE on the grid
    log_density = kde.score_samples(xy_grid_scaled)
    density = np.exp(log_density).reshape(x_grid.shape)

    # Find coords of max density
    max_index = np.unravel_index(np.argmax(density), density.shape)
    x_max_density_scaled, y_max_density_scaled, _ = xy_grid_scaled[np.argmax(density)]

    # Unscale:
    x_max_density, y_max_density, _ = scaler.inverse_transform(np.column_stack([x_max_density_scaled, y_max_density_scaled, np.zeros((1))]))[0, :]

    return x_max_density, y_max_density


def get_nearest_fiber(graph, query_point):
    """
    Finds the nearest fiber in the given graph to the specified query point using a KDTree.

    :param graph: The graph containing vertex positions (vp.v_pos) and vertex labels (vp.v_label).
    :param query_point: A point in the space (e.g., [x, y]) to which the nearest fiber is sought.
    :return: The label of the vertex that corresponds to the nearest fiber.

    Note: The function prints information about the closest coordinate including the distance, index, and label.
    """

    # Convert the vertex positions to a NumPy array
    pos = np.array(list(graph.vp.v_pos))
    
    # Create a KDTree from the positions
    tree = KDTree(pos)
    
    # Query the KD-tree to find the index of the closest coordinate
    distance, index = tree.query(query_point)
    
    # Get the closest coordinate from the index
    closest_coordinate = pos[index]
    
    # Mistake: The variable 'point' is not defined, it should be 'query_point'
    print(f"The closest coordinate to {query_point} is {closest_coordinate} with index={index}, \
    and label {graph.vp.v_label[index]}")

    return graph.vp.v_label[index]


def remove_actin_negative_fibers(actin_img, masks):
    """
    Use the actin image data to remove false positives via thresholding. 
    """
    
    assert actin_img.shape == masks.shape, 'Image and masks dimension do not match'
    
    # get threshold mask
    thresh = threshold_minimum(actin_img)
    myofiber_pixels = actin_img >= thresh

    # mask out incorrect myofiber masks
    masked_mask = np.where(myofiber_pixels, masks, np.zeros_like(masks))

    # reorder the masks
    masked_mask = fastremap.renumber(masked_mask, in_place=False)[0]
    
    return masked_mask


def read_csv_to_dict(filename):
    with open(str(filename), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # Convert CSV rows into dictionary format
        data_dict = {row['Image file name']: int(row['fiber label']) for row in reader}
    return data_dict


def apply_func(data, func, *args, **kwargs):
    out_list = []
    for row_data in tqdm(data,
                         desc=f"Looping over data using {func.__name__}",
                         position=0,
                         leave=True,
                         ascii=True):
        try:
            rd = func(row_data, *args, **kwargs)
        except QhullError:
            rd = 0
        out_list.append(rd)
    return out_list


def get_graph_W(in_graph):
    this_graph_dict = {}
    for node in in_graph.nodes():
        this_graph_dict[node] = list(in_graph.neighbors(node))

    this_graph_w = W(neighbors=this_graph_dict)
    return this_graph_w


def calculate_LISA(adata, feature_name, normalize=True, p_crit=0.05, neighbors_key='spatial_neighbors', update_adata=True):

    adj_mat_key = adata.uns[neighbors_key]['connectivities_key']
    graph = nx.Graph(adata.obsp[adj_mat_key])
    this_graph_w = get_graph_W(graph)

    feature_arr = adata[:, [feature_name]].to_df().values.copy()
    this_graph_moran = Moran(feature_arr, this_graph_w)
    print(f"{this_graph_moran.I=:.2f}, {this_graph_moran.p_sim=}")
    if normalize:
        feature_arr -= feature_arr.mean()
    lag = lag_spatial(this_graph_w, feature_arr)
    this_graph_moran = Moran_Local(feature_arr, this_graph_w)

    labels = {0:'ns',
              1:'HH',
              2:'LH',
              3:'LL',
              4:'HL',
             }

    q_labels_coded = this_graph_moran.q *(this_graph_moran.p_sim < p_crit)
    q_labels = list(map(lambda l: labels[l], q_labels_coded))

    lisa_key = 'LISA-' + feature_name
    if update_adata:
        adata.obs[lisa_key] = q_labels

    return q_labels_coded, q_labels


def retain_largest_component(adata, neighbors_key='spatial_neighbors'):

    # Get the key from .uns?

    G = nx.Graph(adata.obsp[neighbors_key].tocsr())

    # Find the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)

    # Get the adjacency matrix of the original graph
    adj_matrix = nx.adjacency_matrix(G)

    # Create a mask that will zero out entries not in the largest CC
    # Nodes are sorted in the same order as rows/cols in the adjacency matrix
    nodes = list(G.nodes())
    mask = np.isin(nodes, list(largest_cc))

    # Use broadcasting to zero out rows and columns not part of the largest CC
    adj_matrix_largest_cc = adj_matrix.multiply(mask[:, np.newaxis]).multiply(mask)

    # dense_adj_matrix_largest_cc is now a dense matrix representation of the adjacency matrix
    # for the largest connected component, with the same shape as the original and isolated nodes included
    adata.obsp[neighbors_key] = adj_matrix_largest_cc.todense()


def drop_unconnected_rows(adata, neighbors_key='spatial_neighbors'):

    # TODO: get ket from .uns?

    # Assuming adata.obsp[neighbors_key] is a sparse matrix and we sum over its rows
    connected_nodes = np.array(adata.obsp[neighbors_key].sum(axis=1) != 0).flatten()

    # Indices of rows that are connected (i.e., have non-zero sum)
    idx_to_keep = np.where(connected_nodes)[0]

    # Ensure adata is sliced correctly by only selecting rows with indices in idx_to_keep
    adata = adata[idx_to_keep, :].copy()

    return adata


def normalize(vec):
    vec_out = vec - vec.mean()
    vec_out /= vec_out.std()
    return vec_out


def repeat_message_passing(num, adj_arr, inpt):
    prod = np.dot(adj_arr, inpt)
    prod = normalize(prod)
    if num - 1 <= 0:
        res = prod
    else:
        res = repeat_message_passing(num - 1, adj_arr, prod)
    return res


def get_monotonic_cluster_mapping(cluster_labels, size_v):
    c_medians = {}
    for i in np.unique(cluster_labels):
        c_medians[i] = np.median(size_v[cluster_labels == i])
    sorted_clusters = sorted(c_medians.items(), key=lambda x:x[1])
    mapping = [c[0] for c in sorted_clusters]
    cluster_mapping = {c:i for i, c in enumerate(mapping)}
    return cluster_mapping


def fix_message_passing_cluster_labels(series, size_v):
    labels = series.values
    c_map = get_monotonic_cluster_mapping(labels, size_v)
    return series.map(c_map)


def get_MP_clusters(adata_i, message_passing_steps=1, num_clusters=5, random_state=42):
    clusterer = KMeans(n_clusters=num_clusters, random_state=random_state)


    adjacnecy_array = adata_i.obsp['delaunay_connectivities']
    if not isinstance(adjacnecy_array, np.ndarray):
        adjacnecy_array = adjacnecy_array.toarray()

    size_array = adata_i.X[:, adata_i.var_names == 'minFD'].flatten()
    size_array_norm = normalize(size_array)

    cluster_labels = clusterer.fit_predict(repeat_message_passing(message_passing_steps, adjacnecy_array, size_array_norm)[..., np.newaxis])

    key_string = '_'.join(['MP_clustering', str(message_passing_steps)+'step', str(num_clusters)+'c'])
    adata_i.obs[key_string] = cluster_labels
    adata_i.obs[key_string] = fix_message_passing_cluster_labels(adata_i.obs[key_string], size_array)

    return adata_i


def get_subset_distance_distribution(subset_name, subset_nodes, G):
    subset_set = set(np.asarray(subset_nodes).tolist())
    node_dists = {}
    for node in subset_nodes:
        dists = compute_node_distances(node, subset_nodes, G)
        node_dists[node] = dists
    return subset_name, node_dists


def get_MP_clustering_subset_dists(ad_i):

    adj_mat_key = ad_i.uns['delaunay_neighbors']['connectivities_key']
    G = nx.Graph(ad_i.obsp[adj_mat_key])

    subsets_list = [np.where(ad_i.obs['MP_clustering_1step_5c'] == i)[0] for i in range(5)]

    all_subset_dists = {}

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_subset_distance_distribution, subset_str, subset, G) for subset, subset_str in zip(subsets_list, list(range(5)))]
        for future in tqdm(as_completed(futures), total=len(futures), desc='finding subset distance distributions'):
            subset_str, node_distance_dict = future.result()
            all_subset_dists[subset_str] = np.array(list(node_distance_dict.values()))
    return all_subset_dists


def compute_node_distances(node, subset_nodes, G):
    """Find the average shortest path length between `node` and all other `subset_nodes` using a graph, `G`"""
    this_node_path_dict = nx.single_source_shortest_path_length(G, node)
    this_node_path_dict = {node: distance for node, distance in this_node_path_dict.items() if node in subset_nodes}
    this_node_dists = np.array(list(this_node_path_dict.values()))
    return this_node_dists.mean()

