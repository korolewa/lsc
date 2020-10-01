from math import sqrt
from numpy import array, zeros, fill_diagonal, cross, dot
from numpy.linalg import norm
from pymatgen import Element
from ase.data import atomic_numbers, covalent_radii
from itertools import combinations
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components as conn_comps_sci
from networkx.convert_matrix import from_numpy_matrix
from networkx.algorithms.components.connected import connected_components as conn_comps_netx


def adjacency_matrix(structure):
    dist_matrix = structure.distance_matrix
    adj_matrix = zeros(dist_matrix.shape)
    sites = [site for site in structure]

    for i, j in combinations(range(len(structure)), 2):
        if dist_matrix[i][j] < 6.1:
            element_1, element_2 = Element(sites[i].specie.symbol), Element(sites[j].specie.symbol)
            
            if element_1.is_metal or element_1.is_metalloid or element_2.is_metal or element_2.is_metalloid:
                max_distance = covalent_radii[atomic_numbers[element_1.symbol]] + \
                covalent_radii[atomic_numbers[element_2.symbol]] + 0.85
            else:
                max_distance = covalent_radii[atomic_numbers[element_1.symbol]] + \
                covalent_radii[atomic_numbers[element_2.symbol]] + 0.25
                
            if dist_matrix[i][j] < max_distance:
                adj_matrix[i][j] = 1

    fill_diagonal(adj_matrix, 0)

    return array(adj_matrix, dtype=int)


def get_translations(structure, structural_type='100'):
    assert structural_type in ['100', '110']
    
    metal = [Element.from_Z(z).symbol for z in set(structure.atomic_numbers)
             if Element.from_Z(z).is_metal or Element.from_Z(z).is_metalloid]

    mul_structures, conn_components_, ab_indices = [], [], [0, 1, 2]
    conn_indices = [[2, 1, 1], [1, 2, 1], [1, 1, 2]]

    number_connected_components = [conn_comps_sci(adjacency_matrix(structure.__mul__(i)))[0] for i in conn_indices]
    c_index = number_connected_components.index(max(number_connected_components))
    ab_indices.remove(c_index)
    
    extended_structure = structure.__mul__(3)
    extended_components = list(conn_comps_netx(from_numpy_matrix(adjacency_matrix(extended_structure))))
    extended_sites = [[extended_structure[i] for i in components] for components in extended_components]
    layers = [s for s in extended_sites if metal[0] in [site.specie.symbol for site in s]]
        
    max_coords = [max([a.coords[c_index] for a in layer if a.specie.symbol == metal[0]]) for layer in layers]
    first_layer_index = max_coords.index(sorted(max_coords)[0])
    second_layer_index = max_coords.index(sorted(max_coords)[1])

    first_layer_coords = array([a.coords for a in layers[first_layer_index] if a.specie.symbol == metal[0]])
    second_layer_coords = array([a.coords for a in layers[second_layer_index] if a.specie.symbol == metal[0]])
    
    if structural_type == '110':
        first_layer_coords = first_layer_coords[first_layer_coords[:, c_index].argsort()][:int(
            first_layer_coords.shape[0] / 2), :]
        second_layer_coords = second_layer_coords[second_layer_coords[:, c_index].argsort()][:int(
            second_layer_coords.shape[0] / 2), :]        

    a_axis = extended_structure.lattice.matrix[ab_indices][0]
    b_axis = extended_structure.lattice.matrix[ab_indices][1]
    perp = cross(a_axis / norm(a_axis), b_axis / norm(b_axis))
    
    dir_1 = sorted(
        [c[0] - c[1] for c in combinations(first_layer_coords, 2) if abs(
            dot(c[0] - c[1], perp) / norm(c[0] - c[1]) / norm(perp)) < 0.07], key=norm
    )[0]
    m_dist = norm(dir_1)
    dir_1 = dir_1 / norm(dir_1)
    dir_2 = cross(perp / norm(perp), dir_1 / norm(dir_1))
    
    a_projections, b_projections = [], []
    
    for site_coords in first_layer_coords:
        nearest_site = second_layer_coords[KDTree(second_layer_coords).query(site_coords)[1]]
        a_projections.append(dot((site_coords - nearest_site), dir_1))
        b_projections.append(dot((site_coords - nearest_site), dir_2))

    a_translation = min([min(abs(m_dist - abs(p) % m_dist), abs(p) % m_dist) for p in a_projections]) / m_dist
    
    if structural_type == '110':
        m_dist = m_dist * sqrt(2)    
    
    b_translation = min([min(abs(m_dist - abs(p) % m_dist), abs(p) % m_dist) for p in b_projections]) / m_dist
    
    return sorted([round(a_translation, 2), round(b_translation, 2)])
