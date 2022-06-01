#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions to manipulate pyvista meshes.

@author: anbieber
"""
import numpy as np
import matplotlib.pyplot as plt
from pyvistaqt import BackgroundPlotter
from scipy.spatial import cKDTree, distance_matrix
import pyvista as pv

from utility_functions import normalize_vector, angle_between_vectors, angle_between_vectors_batch

#%% Small utility functions

def get_mean_cell_area(mesh):
    """Get the mean cell area of a pyvista mesh."""
    tmp = mesh.compute_cell_sizes(area=True, length=False, volume=False)
    return(np.mean(tmp['Area']))

def n_connected_regions(mesh):
    """Get the number of connected regions in a pyvista mesh, based on the parameter 'RegionId'."""
    tmp = mesh.connectivity()
    return len(np.unique(tmp['RegionId']))

#%% Simple mesh manipulation functions
    
def find_vertices_cell_ids(surf, cell_ids):
    """For a given subset of cell ids of a triangular mesh, find ids of all vertices of those cells."""
    # Get faces of surface
    faces = surf.faces.reshape(-1,4)[:,1:4]
    return np.unique(faces[cell_ids,:].flatten())

def extract_mesh_cell_ids(mesh, cell_ids, keep_cells=True):
    """
    Extract a subset of a mesh based on given cell ids.
    
    The cells indicated are retained if keep_cells=True, otherwise the 
    full mesh is returned with the indicated cells cut out.
    """
    if keep_cells:
        cell_ids_to_delete = np.delete( np.arange(mesh.n_cells), cell_ids )
        cell_ids_to_keep = cell_ids
    else:
        cell_ids_to_delete = cell_ids
        cell_ids_to_keep = np.delete( np.arange(mesh.n_cells), cell_ids )
        
    vertices_to_delete = find_vertices_cell_ids(mesh, cell_ids_to_delete)
    vertices_to_keep = find_vertices_cell_ids(mesh, cell_ids_to_keep)
    same_ids, id_delete, id_keep = np.intersect1d(vertices_to_delete, vertices_to_keep, return_indices=True)
    vertices_to_delete_final = np.delete(vertices_to_delete, id_delete)
    
    mesh_out, _ = mesh.remove_points(vertices_to_delete_final, mode='any', keep_scalars=True, inplace=False)
    
    return mesh_out

#%% Functions for creating new meshes

def make_2d_grid_faces(size):
    """Get vertex ids of all faces of a regular 2d grid with the given size (x_size, y_size)."""
    n_vertices = size[0]*size[1]
    ids_all_verts = np.arange(n_vertices).reshape(size)
    # Get vertex ids of upper triangles
    ids_up = np.c_[ids_all_verts[:-1,:-1].flatten(), ids_all_verts[:-1,1:].flatten(), ids_all_verts[1:, :-1].flatten()]
    # Get vertex ids of lower triangles
    ids_low = np.c_[ids_up[:,1:], ids_all_verts[1:,1:].flatten()]
    # Make a big faces array
    faces = np.empty((2*ids_up.shape[0], 4))
    faces[:,0] = 3 # first value is number of vertices: all triangles!
    faces[0::2,1:] = ids_up
    faces[1::2,1:] = ids_low
    return faces.astype(int)
    
def triangulate_ladder(a,b):
    """Given two rows of a and b sorted along the z axis, make a mesh filling the area between all points."""
    # Make sure that a and b are both sorted along z
    a = a[np.argsort(a[:,2])]
    b = b[np.argsort(b[:,2])]
    # Find nearest neighbors of a in b and b in a
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    _, idx_ba = tree_a.query(b)
    _, idx_ab = tree_b.query(a)
    # Concatenate points and get their ids
    points = np.r_[a,b]
    a_ids = np.arange(len(a))
    b_ids = np.arange(len(b)) + len(a)
    # Find all pairs of a and b and sort them
    ab_pairs_0 = np.r_[ np.c_[a_ids, b_ids[idx_ab]], np.c_[idx_ba, b_ids] ]
    ab_pairs = np.sort(np.unique(ab_pairs_0, axis=0))
    faces = []
    previous_row = ab_pairs[0]
    for row in ab_pairs[1:]:
        if np.any(np.isin(row, previous_row)):
            # If one of the vertices is the same, the face is described by the three unique vertices
            faces.append(np.unique(np.c_[previous_row, row]))
        else:
            # If none of the vertices is the same, find the best middle row that connects them
            middle_rows_pot = np.array([[previous_row[0], row[1]], [row[0], previous_row[1]]])
            middle_row_lengths = np.linalg.norm(points[middle_rows_pot[:,0] - middle_rows_pot[:,1]], axis=1)
            middle_row = middle_rows_pot[np.argmin(middle_row_lengths)]
            faces.append(np.unique(np.c_[previous_row, middle_row]))
            faces.append(np.unique(np.c_[middle_row, row]))
        # Update the previous row
        previous_row = row
    # Build the mesh
    mesh_faces = np.c_[np.repeat(3, len(faces)), np.array(faces)]
    mesh = pv.PolyData(points, mesh_faces)      
    return mesh


#%% Find mesh borders and manipulate them

def mesh_find_edges(surf):
    """Find all edges in a mesh."""
    # If there are faces that are not triangles, triangulate first
    if not surf.is_all_triangles(): 
        surf.triangulate(inplace=True)
    # Get faces of surface
    faces = surf.faces.reshape(-1,4)[:,1:4]
    # Get edges of faces and count their occurence
    edges = np.concatenate((faces[:,0:2], faces[:,1:], faces[:,0::2]), axis=0)
    edges = np.sort(edges, axis=1)
    edges_unique, edge_counts = np.unique(edges, return_counts=True, axis=0)
    return edges_unique, edge_counts, faces
    

def mesh_find_borders(surf, edges_unique=None, edge_counts=None, faces=None):
    """
    Find all borders (open edges) of a mesh.
    
    If unique edges, edge counts and the array of face ids were already calculated, 
    they can be passed as arguments to speed up calculations.
    Returns ids of border vertices and faces and array of border edges.
    To remove the border cells, use surf.remove_points(border_vertices) with the default settings
    mode='any', keep_scalars=True. 
    """
    if not np.all([type(a)==np.ndarray for a in [edges_unique, edge_counts, faces]]):
        # Get edges of faces and count their occurence
        edges_unique, edge_counts, faces = mesh_find_edges(surf)
    # Get border edges, vertices and faces containing these vertices
    border_edges = edges_unique[edge_counts==1]
    border_vertices = np.unique(border_edges)
    border_face_ids = np.where( np.any(np.isin(faces, border_vertices), axis=1) )[0]
    return border_vertices, border_face_ids, border_edges

def mesh_remove_borders_distance(surf, dist=1.5, dist_unit='median_edge_length'):
    """Remove cells of a mesh within a certain distance to its borders.
    
    First finds border vertices and then all vertices within a certain distance to these border vertices.
    The distance cutoff dist is multiplied with a factor calculated depending on dist_unit. 
    Possible entries for dist_unit are 'median_edge_length', 'mean_edge_length', an int, float or None.
    """
    # get edges of mesh
    edges_unique, edge_counts, faces = mesh_find_edges(surf)
    # If needed, calculate median / mean edge length
    if type(dist_unit)==str:
        edge_lengths = np.linalg.norm(surf.points[edges_unique[:,0]] - surf.points[edges_unique[:,1]], axis=1) 
        if dist_unit == 'median_edge_length':
            dist_factor = np.median(edge_lengths)
        elif dist_unit == 'mean_edge_length':
            dist_factor = np.median(edge_lengths)
    elif type(dist_unit) in (float, int):
        dist_factor=dist_unit
    else:
        dist_factor = 1
    # Get initial border vertices
    border_verts0, _, _ = mesh_find_borders(surf, edges_unique=edges_unique, edge_counts=edge_counts, faces=faces)
    # Calculate distances to initial border vertices and apply cutoff
    tree_border = cKDTree(surf.points[border_verts0])
    dist_border, _ = tree_border.query(surf.points)
    ids_to_remove = np.where(dist_border <= dist*dist_factor)[0]
    # Remove the border
    surf_clean,_ = surf.remove_points(ids_to_remove, mode='any', keep_scalars=True, inplace=False)
    return surf_clean, tree_border


    
#%% Finding neighbor vertices and associated values

def mesh_neighbor_id_list(surf):
    """Get ids of all neighboring vertices for each vertex in a mesh.
    
    Currently only works for all-triangle meshes.
    """
    assert surf.is_all_triangles(), "surf is not all triangles!"
    # Get faces of surface
    faces = surf.faces.reshape(-1,4)[:,1:4]
    # initialize an empty list of lists
    list_tmp = [[] for i in range(surf.n_points)]
    # Go through faces and add neighbors to lists
    for face in faces:
        a,b,c = face
        for idx, vals in zip([a,b,c], [[b,c], [a,c], [a,b]]):
            list_tmp[idx]+=vals
            
    # Reduce lists to unique elements
    list_out = [np.unique(row) for row in list_tmp]
    
    return list_out

def mesh_neighbor_cells(surf):
    """Get ids of all neighboring cells for each cell in a mesh.
    
    Currently only works for all-triangle meshes.
    """
    assert surf.is_all_triangles(), "surf is not all triangles!"
    # Get faces of surface
    faces = surf.faces.reshape(-1,4)[:,1:4]
    
    # For each vertex in the mesh, find all faces that contain it ----------------------
    # Initialize an empty list of lists to put for each vertex all adjacent faces
    faces_at_vertices = [[] for i in range(surf.n_points)]
    # Fill the list with face ids
    for i, face in enumerate(faces):
        for v in face:
            faces_at_vertices[v].append(i)
    # For each cell (face) in the mesh, get all its neighboring faces based on faces_at_vertices
    # Initialize an empty list to store results
    faces_at_faces = [[] for i in range(surf.n_cells)]
    # Iterate a second time through faces & combine neighboring faces of all their vertices
    for i, face in enumerate(faces):
        all_local_faces = []
        # For each vertex in the face, find the neighboring faces
        for v in face:
            all_local_faces += faces_at_vertices[v]
        # remove duplicates
        all_local_faces = np.unique(all_local_faces)
        # remove self
        faces_at_faces[i] = np.delete(all_local_faces, np.where(all_local_faces==i)[0])
    return faces_at_faces
        

def mesh_neighbor_further_layers(neighbor_id_list, add_layer=1):
    """Given a list of neighboring ids describing a mesh (vertices or cells), add more layers (neighbor's neighbor etc)."""
    working_list = neighbor_id_list.copy()
    dict_neighbor_lists = {}
    for n in range(add_layer):
        list_tmp = [[] for i in range( len(working_list) )]
        for i,row in enumerate(working_list):
            for entry in row:
                list_tmp[i] +=[idx for idx in neighbor_id_list[entry]]
        # Reduce lists to unique elements
        working_list_0 = [np.unique(row) for row in list_tmp]
        # Remove self index
        working_list = [np.delete(row, np.where(row==i)[0]) for i, row in enumerate(working_list_0)]
        dict_neighbor_lists['layer_{:d}'.format(n+2)] = working_list
    return dict_neighbor_lists

def mesh_get_neighbor_distances(points, neighbor_ids, output='mean'):
    """Get mean, min or max distance between vertices in a mesh."""
    distances = []
    for p_self, ids_neighbors in zip(points, neighbor_ids):
        p_neighbors = points[ids_neighbors]
        if output == 'mean':
            distances.append( np.mean(np.linalg.norm(p_neighbors-p_self, axis=1)) )
        elif output == 'min':
            distances.append( np.min(np.linalg.norm(p_neighbors-p_self, axis=1)) )
        elif output == 'max':
            distances.append( np.max(np.linalg.norm(p_neighbors-p_self, axis=1)) )
    return np.array(distances)

def mesh_get_neighbor_values(values, neighbor_ids):
    """Get mean and difference values associated with the neighboring vertices for each vertex in a mesh."""
    mean_values = np.array([ np.mean(values[ids]) for ids in neighbor_ids ])
    diff_self_neighbors = values - mean_values
    return mean_values, diff_self_neighbors

def mesh_cells_within_radius(cell_xyz, radius):
    """For a list of cell coordinates, get all cell ids within a certain radius.
    
    Note: distance matrix is too slow for medium size meshes. Try p dist!
    """
    # Get distance matrix
    distmat = distance_matrix(cell_xyz, cell_xyz)
    # For each row, find indices where distance <= radius
    result_list = [np.nonzero(row <= radius)[0] for row in distmat]
    return result_list


#%% Normal refinement

def mesh_refine_normals(mesh, neighbor_point_ids, normal_key='Normals_',
                        iterations=1, plot_mesh=False, plot_N_factor=-1):
    """
    Refine mesh normals by substituting each vertex normal with the mean of the neighboring normals.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input pyvista mesh.
    neighbor_point_ids : list of lists
        List of neighbor vertices for each vertex in the mesh.
    normal_key : string, optional
        Array name under which normals are saved in mesh. The default is 'Normals_'.
    iterations : int, optional
        Number of refinement iterations. The default is 1.
    plot_mesh : bool, optional
        Whether to generate a BackgroundPlotter visualization of the mesh. The default is False.
    plot_N_factor : TYPE, optional
        Factor with which normals are multiplied for plotting. The default is -1.

    Returns
    -------
    normals : numpy.ndarray
        Array of refined normals.
    normal_angles_final : numpy.ndarray
        Angles of new normals to original normals.
    p0 : pyvistaqt.BackgroundPlotter, optional
        BackgroundPlotter instance, containing mesh with new normals.
    """
    normals = mesh[normal_key].copy()
    for i in range(iterations):
        new_normals = []
        normal_angles = []
        for orig_N, point_ids in zip(normals, neighbor_point_ids):
            mean_N = normalize_vector( np.sum(normals[point_ids], axis=0) )
            new_normals.append(mean_N)
            normal_angles.append(angle_between_vectors(orig_N, mean_N, normalized=True) )
        print('Iteration {}: mean correction angle is {} degrees.'.format(i,np.mean(normal_angles)))
        normals = np.array(new_normals)

    normal_angles_final = angle_between_vectors_batch(normals, mesh[normal_key])
    
    if plot_mesh:
        p0 = BackgroundPlotter()
        p0.add_mesh(mesh, scalars=normal_angles_final)
        p0.add_scalar_bar(title='Normal correction angle')
        p0.add_arrows(mesh.points, plot_N_factor*mesh[normal_key], color='black', label='orig')
        p0.add_arrows(mesh.points, plot_N_factor*normals, color='white', label='new')
        p0.add_legend()
        
        return normals, normal_angles_final, p0
    
    return normals, normal_angles_final


def mesh_refine_normals_v1(normals, neighbor_point_ids, iterations=1, tolerance_angle=20):
    """
    Refine normals of a mesh.
    
    All normals are refined which have a higher median angle to their neighbors than the tolerance angle.

    Parameters
    ----------
    normals : numpy.ndarray
        Array of normals of the mesh.
    neighbor_point_ids : list of lists
        List of neighbor vertices for each vertex associated with the normals.
    iterations : int, optional
        Number of refinement iterations. The default is 1.
    tolerance_angle : float, optional
        Cutoff angle determining which normals are refined, in degrees. The default is 20.

    Returns
    -------
    new_normals : 
        Array of refined normals.
    """
    for i in range(iterations):
        list_new_normals = []
        normal_angles = []
        n_corrected = 0
        for orig_N, point_ids in zip(normals, neighbor_point_ids):
            neighbor_normals = normals[point_ids]
            neighbor_angles = angle_between_vectors_batch(neighbor_normals, orig_N)
            if np.median(neighbor_angles) < tolerance_angle:
                list_new_normals.append(orig_N)
                normal_angles.append(0)
            else:
                mean_N = normalize_vector( np.sum(neighbor_normals, axis=0) )
                list_new_normals.append(mean_N)
                normal_angles.append(angle_between_vectors(orig_N, mean_N, normalized=True) )
                n_corrected += 1
        print('Iteration {}: Corrected {} normals. '.format(i,n_corrected))
        new_normals = np.array(list_new_normals)
    
    return new_normals


#%% Mesh functions related to hole filling

def group_connected_edges(edges):
    """For an array containing lots of mesh edges, group the vertices to give connected pieces."""
    # Set up the list of pointsets by taking the first edge
    list_pointsets = [set(edges[0])]
    # Iterate through the other edges
    for edge in edges[1:]:
        # Check overlaps with existing pointsets
        s_edge = set(edge)
        set_ids = np.invert( [ s.isdisjoint(s_edge) for s in list_pointsets] ) # Gives True for all sets in which at least one of the new elements appears
        if np.sum(set_ids) == 0:
            # If none of the vertices is in any existing set, make a new set
            list_pointsets.append(s_edge)
        elif np.sum(set_ids) == 1:
            # If one or both of the vertices was found in one set, add vertices to the set (if necessary)
            idx_s = np.where(set_ids)[0][0]
            list_pointsets[idx_s] = list_pointsets[idx_s].union(s_edge)
        else:
            # If the two vertices were found in two different sets, join these sets (no need to add in vertices since they're already there)
            ids_ab = np.sort( np.where(set_ids)[0] )
            s_move = list_pointsets.pop(ids_ab[1])
            list_pointsets[ids_ab[0]] = list_pointsets[ids_ab[0]].union( s_move )
    return list_pointsets

def mesh_get_connected_border_loops(mesh, plot_loops=False):
    """Given a mesh, find connected loops of its borders."""
    # Get border vertices
    _, _, border_edges = mesh_find_borders(mesh)    
    border_vert_unique, border_vert_count = np.unique(border_edges.flatten(), return_counts=True)
    # Get vertices of loops: all the vertices should appear two times
    border_vert_double = border_vert_unique[border_vert_count == 2]    
    border_connected_edges = np.array( [edge for edge in border_edges if np.logical_and(edge[0] in border_vert_double, edge[1] in border_vert_double)])
    border_end_edges = np.array( [edge for edge in border_edges if np.logical_xor(edge[0] in border_vert_double, edge[1] in border_vert_double)])
    border_end_set = set( list(border_end_edges.flatten()) )
    # Group the vertices into connected sets
    connected_pointsets = group_connected_edges(border_connected_edges)
    
    # For each connected pointset, test if it has any elements in common with the end pieces
    list_loop_sets = []
    dict_open_borders = {}
    for i, s in enumerate( connected_pointsets ):
        # A closed loop should not have overlaps with the edge vertices
        if s.isdisjoint(border_end_set):
            list_loop_sets.append(s)
        else :
            dict_open_borders[i] = {'inner_point_set': s,
                                    'outmost_inner_points': s.intersection(border_end_set)}
            s_edge = list(dict_open_borders[i]['outmost_inner_points'])
            dict_open_borders[i]['outer_edges'] = [edge for edge in border_end_edges if len(np.intersect1d(edge, s_edge)) > 0]
    
    # Plot loops if desired
    if len(list_loop_sets) == 0:
        print('Did not find any closed loops!')
        plot_loops = False
        
    if plot_loops:
        p0 = BackgroundPlotter()
        p0.enable_eye_dome_lighting()
        p0.add_mesh(mesh, color='white', show_edges=False)
        cmap = plt.get_cmap('tab10')
        for i, s in enumerate(list_loop_sets):
            p0.add_mesh(mesh.points[list(s)], color=cmap(i), label='set {}'.format(i))  
        p0.add_legend()

        
    return list_loop_sets, dict_open_borders, border_connected_edges

def sort_connected_edges(edges):
    """For a list of sets of edges which describe a closed loop, sort them and their vertices so one can follow the loop once around."""    
    list_edges = [ edges[0] ] # initiate the list of edges, starting with the first (this is a random choice)
    set_local = edges[0].copy() # Set local is a set containing all vertices that have been added. This is to identify the new vertex when adding a new set
    edges_local = edges.copy() # Edges local: a copy of the original list, allows deleting items whenever they've been added
    edges_local.pop(0) # Since edges[0] have already been added, get rid of them
    list_vertices = list(edges[0]) # initiate the list of sorted vertices
    next_vertex = list_vertices[1] # Take one of the vertices to search for the next edge. To keep the order, we need to take the one at index 1.
    
    for i in range(len(edges)-1):
        next_id = np.where([next_vertex in s for s in edges_local])[0][0] # Find id of the next edge
        set_add = edges_local[next_id] # Extract the next edge
        list_edges.append(set_add) # Add this edge to the list of edges
        if len(edges_local) == 1: # If the list of edges is down to one (the edge just added was the last one, don't bother doing the rest)
            continue
        next_vertex = list(set_add.difference(set_local))[0] # Find the next vertex to search for: the vertex that's not yet in set_local
        list_vertices.append(next_vertex) # Save that vertex in the list of vertices
        set_local = set_local.union(set_add) # Update set_local by adding the edge
        edges_local.pop(next_id) # Remove the edge from the list

    return list_edges, list_vertices

def sort_connected_edges_open(edge_array):
    """For an array of edges which describe an open chain, sort them and their vertices so one can follow the loop once around."""    
    vert_unique, vert_counts = np.unique(edge_array.flatten(), return_counts=True)
    start_vert = vert_unique[vert_counts==1][0]
    start_edge_id = np.where(edge_array==start_vert)[0][0]  
    start_edge = edge_array[start_edge_id]
    
    # Basically same code as in connected case. Turn edges into sets!   
    edges = [set(list(row)) for row in edge_array]
    list_edges = [ edges[start_edge_id] ] # initiate the list of edges, starting with the start edge
    set_local = edges[start_edge_id].copy() # Set local is a set containing all vertices that have been added. This is to identify the new vertex when adding a new set
    edges_local = edges.copy() # Edges local: a copy of the original list, allows deleting items whenever they've been added
    edges_local.pop(start_edge_id) # Since start edge vertices have already been added, get rid of them

    # Get next vertex and initiate list of sorted vertices
    next_vertex = start_edge[start_edge != start_vert][0]
    list_vertices = [start_vert, next_vertex] # initiate the list of sorted vertices
    
    for i in range(len(edges)-1):
        next_id = np.where([next_vertex in s for s in edges_local])[0][0] # Find id of the next edge
        set_add = edges_local[next_id] # Extract the next edge
        list_edges.append(set_add) # Add this edge to the list of edges
        #if len(edges_local) == 1: # If the list of edges is down to one (the edge just added was the last one, don't bother doing the rest)
        #    continue
        next_vertex = list(set_add.difference(set_local))[0] # Find the next vertex to search for: the vertex that's not yet in set_local
        list_vertices.append(next_vertex) # Save that vertex in the list of vertices
        set_local = set_local.union(set_add) # Update set_local by adding the edge
        edges_local.pop(next_id) # Remove the edge from the list

    return list_edges, list_vertices

#%% Functions for removing too small cells in meshes
        
def mesh_merge_point_ids(mesh, ids_keep, ids_remove):
    """Given an all-triangle pyvista mesh, merge the points with ids_remove into the points given by ids_keep.
    
    Returns an all-triangle mesh; the array 'original_point_ids' can be used to transfer values.
    If an id is given two times in ids_remove, it's currently skipped the second time.
    """
    # Get faces of surface
    assert mesh.is_all_triangles(), 'mesh_merge_point_ids only works for all triangle meshes!'
    faces = mesh.faces.reshape(-1,4)[:,1:4]
    # To tackle the problem of changing ids, keep an array of all point ids where ids are changed
    all_point_ids_orig = np.arange(mesh.n_points)
    all_point_ids = all_point_ids_orig.copy()
    # Iterate through pairs of ids that should be merged.
    removed_ids = []
    for i0, i1 in zip(ids_keep, ids_remove):
        if i1 in removed_ids:
            continue
        # Replace i1 with i0 in all faces, then remove the faces that now have 2x i0
        faces[faces==i1] = i0
        faces_del_ids = np.where(np.sum(np.isin(faces, i0), axis=1) == 2)[0]
        faces = np.delete(faces, faces_del_ids, axis=0)
        # Update ids: all ids that were originally > i1 : -1
        all_point_ids[all_point_ids_orig > i1] -= 1
        # Collect the processed ids to avoid double processing
        removed_ids.append(i1)
    # Update the point ids in the faces
    new_faces = np.c_[np.repeat(3,faces.shape[0]), all_point_ids[faces]]
    # Get points and their original id
    orig_point_ids = np.delete(np.arange(mesh.n_points), removed_ids)
    new_points = mesh.points[orig_point_ids]
    # Make the new mesh
    mesh_new = pv.PolyData(new_points, new_faces)
    mesh_new['original_point_ids'] = orig_point_ids
    return mesh_new

    
def mesh_fuse_edges(mesh, edges, edge_ids_to_fuse=None, vertices_to_keep=None):
    """Given a mesh and edges in this mesh, fuse the edges indicated by the ids, keeping the vertices indicated."""
    if np.all(edge_ids_to_fuse == None):
        edge_ids_to_fuse = np.arange(len(edges))
    if np.all(vertices_to_keep == None):
        vertices_to_keep = np.array([])
    ids_keep = []
    ids_remove = []
    processed_ids = set([])
    for i in edge_ids_to_fuse:
        s_edge = set(edges[i])
        ids = np.array( list(edges[i]) )

        if s_edge.isdisjoint(processed_ids):
            ids_to_keep = np.isin(ids, vertices_to_keep)
            if np.sum(ids_to_keep) == 0:
                ids_to_keep = np.array([True, False])
            elif np.sum(ids_to_keep) == 2:
                continue
        elif s_edge.issubset(processed_ids):
            continue
        else :
            # Find the point that was already processed
            id_processed = list(s_edge.intersection(processed_ids))[0]
            if id_processed in ids_keep:
                ids_to_keep = ids==id_processed
            else :
                ids_to_keep = ids!=id_processed
                            
        ids_keep.append(ids[ids_to_keep][0])
        ids_remove.append(ids[ids_to_keep==False][0])
        processed_ids = processed_ids.union(s_edge)
    
    mesh1 = mesh_merge_point_ids(mesh, ids_keep, ids_remove)
    return mesh1

def clean_mesh_borders_after_clipping(mesh, orig_mesh, min_edge_ratio=0.1):
    """After clipping a mesh, clean up the edge, removing too short edges by fusing points."""
    # Get original edge lengths to determine cutoff length
    orig_edges, _, _ = mesh_find_edges(orig_mesh)
    orig_edge_lengths = np.linalg.norm(orig_mesh.points[orig_edges[:,0]] - orig_mesh.points[orig_edges[:,1]], axis=1 )
    orig_edge_median_length = np.median(orig_edge_lengths)
    min_edge_length = min_edge_ratio*orig_edge_median_length
    # Check if clipped mesh is triangles only
    if not mesh.is_all_triangles():
        mesh = mesh.triangulate()
    # Find the new points after clipping (along the clipping surface)
    clip_point_ids = np.where(np.sum(np.isin(mesh.points, orig_mesh.points), axis=1) < 3)[0]
    #clip_new_points = mesh.points[clip_point_ids]
    
    # Get faces & edges at the new border of clipped mesh & sort them
    edges, edge_count, faces = mesh_find_edges(mesh)
    border_edges = edges[edge_count==1]
    new_border_edges = border_edges[np.where(np.all(np.isin(border_edges, clip_point_ids), axis=1))[0]]
    new_old_connections = border_edges[np.where(np.sum(np.isin(border_edges, clip_point_ids), axis=1)==1)[0]]
    outer_vertices = np.intersect1d(new_border_edges.flatten(), new_old_connections.flatten())
    
    # Prepare for cleaning: keep the outermost vertices, and fuse all edges that are too short
    new_border_lengths = np.linalg.norm(mesh.points[new_border_edges[:,0]] - mesh.points[new_border_edges[:,1]], axis=1)
    edge_ids_to_process = np.where(new_border_lengths < min_edge_length)[0]
    # Generate the intermediate mesh    
    mesh1 = mesh_fuse_edges(mesh, new_border_edges, edge_ids_to_process, outer_vertices)
    return mesh1

def clean_mesh_small_faces(mesh1, min_cell_area):
    """Find cells in a mesh smaller than the min_cell_area, and remove them by fusing one of their edges.
    
    The border of the mesh is preserved. If any of the small cells is in a corner 
    (and thus has only border vertices), this cell is kept and not changed.
    """
    areas = mesh1.compute_cell_sizes(length=False, area=True, volume=False)['Area']
    faces = mesh1.faces.reshape(-1,4)[:,1:]
    # Find the cells with too small areas that need to be removed
    faces_small = faces[areas < min_cell_area]
    # Find border vertices --> these should be kept
    border_vertices, _,_ = mesh_find_borders(mesh1)
    
    # In each face, find the shortest edge and second shortest edge
    faces_small_edges = np.dstack( (faces_small[:,0:2], faces_small[:,1:], faces_small[:,0::2]) )
    faces_small_edge_lengths = np.linalg.norm( mesh1.points[faces_small_edges[:,0,:] ] - mesh1.points[faces_small_edges[:,1,:] ], axis=-1 )
    edge_ids_sortlength = np.argsort(faces_small_edge_lengths, axis=1)
    edges_short_0 = np.array( [np.sort(edges[:,i]) for edges, i in zip(faces_small_edges, edge_ids_sortlength[:,0])] )
    edges_short_1 = np.array( [np.sort(edges[:,i]) for edges, i in zip(faces_small_edges, edge_ids_sortlength[:,1])] )
    # Assemble edges to be removed:
        # First choice is the shortest edge
        # If this is a border edge, try second shortest edge
    edges_to_remove_all = edges_short_0.copy()
    edges_in_border = np.all( np.isin(edges_to_remove_all, border_vertices), axis=1)
    ids_to_remove = []
    if np.any(edges_in_border):
        for idx in np.where(edges_in_border)[0]:
            if np.all(np.isin(edges_short_1[idx], border_vertices)):
                # If all points are border vertices, do not touch this cell
                ids_to_remove.append(idx)
                # Only remove in the end so ids don't get messed up
                #edges_to_remove_all = np.delete(edges_to_remove_all, idx, axis=0)
                print('Small face at corner, skipping..')
            else:
                edges_to_remove_all[idx] = edges_short_1[idx]
    # Remove the edges to remove
    edges_to_remove_all = np.delete(edges_to_remove_all, ids_to_remove, axis=0)
    # Condense to unique edges            
    edges_to_remove = np.unique(edges_to_remove_all, axis=0)
    # Remove these edges by fusing the vertices
    mesh2 = mesh_fuse_edges(mesh1, edges_to_remove, edge_ids_to_fuse=None, vertices_to_keep=border_vertices)
    
    return mesh2


def clean_mesh_after_clipping_v1(mesh, orig_mesh, border_min_ratio=0.3, area_min_ratio=0.1, 
                                 transfer_arrays=True, debug=False):
    """After clipping orig_mesh to give mesh, clean up the result.
    
    Cleaning is performed in two steps:
        1. Remove points in the new border that are too close to each other
        2. Merge faces that are too small, keeping the border points constant.
    The ratios are cutoff ratios used on the median of the respective value in the original mesh.
    """
    # Find the new border and fuse border points that are too close together
    mesh1 = clean_mesh_borders_after_clipping(mesh, orig_mesh, min_edge_ratio=border_min_ratio)
    # Second cleaning steps: remove cells that are too small by fusing them along the shortest edge
    # Get original smallest area in the mesh --> cutoff
    orig_areas = orig_mesh.compute_cell_sizes(length=False, area=True, volume=False)['Area']
    min_cell_area = area_min_ratio*np.median(orig_areas)
    mesh2 = clean_mesh_small_faces(mesh1, min_cell_area)
    # Since this is the second cleaning step, update the original point ids  
    orig_ids = mesh1['original_point_ids'][mesh2['original_point_ids']]
     
    # Transfer values of arrays if wanted
    if transfer_arrays and mesh.n_arrays > 0:
        for name in mesh.array_names:
            mesh2[name] = mesh[name][orig_ids].copy()    
    mesh2['real_orig_point_ids'] = orig_ids
    if debug:
        return mesh2, mesh1, mesh, orig_ids   
      
    return mesh2





