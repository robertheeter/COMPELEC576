'''
dataset.py

ABOUT:
- 
'''
import os
import numpy as np
import pandas as pd
import scipy as sp
import networkx as nx
import multiprocessing
import torch
import torch_geometric as pyg

# from torch_geometric.utils.convert import from_networkx
from rdkit import Chem
from log.logger import Logger

# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


'''
one_hot:
    about:
        function to assist with one-hot encoding
    params:
        val = value
        set = set of possible values
    returns:
        one-hot vector with index of val set to 1
'''
def one_hot(val, set):

    if val not in set:
        val = set[0]
    return list(map(lambda s: val == s, set))


'''
get_nodes:
    about:
        gets node and node feature tensor for protein chain using RDKit to determine relevant atom chemical features
    params:
        protein_chain = RDKit mol data file for protein chain
    returns:
        x = torch tensor of node and node features
'''
def get_nodes(protein_chain):

    G = nx.Graph()

    # iterate over all atoms in protein and calculate/assign features
    atoms = protein_chain.GetAtoms()
    
    for i in range(len(atoms)):

        atom = atoms[i]
        atom_index = atom.GetIdx()

        # element symbol [other, C, N, O, P, S, F, CL, Br, I]
        element_symbol = one_hot(atom.GetSymbol(), ['Other','C','N','O','P','S'])
        
        # degree [0, 1, 2, 3, 4, 5, 6, 7]
        degree = one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7])

        # formal charge [0, 1, 2, -1, -2]
        formal_charge = one_hot(atom.GetFormalCharge(), [0, 1, 2, -1, -2])
        
        # radical electrons [0, >=1]
        radical_electrons = one_hot(atom.GetNumRadicalElectrons(), [1, 0])

        # implicit valence [0, 1, 2, 3, 4, 5, 6]
        implicit_valence = one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])

        # implict hydrogens [0, 1, 2, 3, 4]
        implicit_hydrogens = one_hot(atom.GetNumImplicitHs(), [0, 1, 2, 3, 4])

        # hybridization [SP, SP2, SP3, SP3D, SP3D2]
        hybridization = one_hot(atom.GetHybridization(), [Chem.HybridizationType.SP,
                                                          Chem.HybridizationType.SP2,
                                                          Chem.HybridizationType.SP3,
                                                          Chem.HybridizationType.SP3D,
                                                          Chem.HybridizationType.SP3D2])

        # aromatic [0/1]
        aromatic = atom.GetIsAromatic()

        # combine all 39 features into one vector and add to graph
        features = np.hstack((element_symbol, degree, formal_charge, radical_electrons, implicit_valence, implicit_hydrogens, hybridization, aromatic))

        G.add_node(atom_index, feats=torch.from_numpy(features))

    x = torch.stack([feats['feats'] for n, feats in G.nodes(data=True)]).float()

    return x


def get_edges(protein_chain, radius_ncov):

    G = nx.Graph()

    pos = protein_chain.GetConformers()[0].GetPositions()
    dist_matrix = sp.spatial.distance_matrix(pos, pos)

    for bond in protein_chain.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        G.add_edge(i, j, type=0, dist=dist_matrix[i, j])

    node_idx = np.where((dist_matrix <= radius_ncov))
    for i, j in zip(node_idx[0], node_idx[1]):
        i = int(i)
        j = int(j)

        # omit covalent edges (to prevent double-counting of edges)
        if (not protein_chain.GetBondBetweenAtoms(i, j)) and (i != j):
            G.add_edge(i, j, type=1, dist=dist_matrix[i, j])
    

    G = G.to_directed()

    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in G.edges(data=False)]).T
    edge_attr = torch.stack([torch.FloatTensor((a['type'], a['dist'])) for _, _, a in G.edges(data=True)])

    return edge_index, edge_attr


'''
get_edges_cov:
    about:
        gets covalent edge tensor for protein chain using protein covalent bonds
    params:
        protein_chain = RDKit mol data file for protein chain
    returns:
        edge_index_cov = torch tensor of covalent edges
        edge_attr_cov = torch tensor of covalent edge attributes
'''
# def get_edges_cov(protein_chain):
    
#     G_cov = nx.Graph()

#     pos = protein_chain.GetConformers()[0].GetPositions()
#     dist_matrix = sp.spatial.distance_matrix(pos, pos)

#     for bond in protein_chain.GetBonds():

#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()
        
#         G_cov.add_edge(i, j, type=0, dist=dist_matrix[i, j])
    
#     G_cov = G_cov.to_directed()

#     edge_index_cov = torch.stack([torch.LongTensor((u, v)) for u, v in G_cov.edges(data=False)]).T
#     edge_attr_cov = torch.stack([torch.FloatTensor((a['type'], a['dist'])) for _, _, a in G_cov.edges(data=True)])

#     return edge_index_cov, edge_attr_cov


'''
get_edges_ncov:
    about:
        gets non-covalent edge tensor for protein chain between a given atom and its neighbors within radius_ncov
    params:
        protein_chain = RDKit mol data file for protein chain
        radius_ncov = cutoff radius for determining non-covalent edges around an atom in Angstroms
    returns:
        edge_index_ncov = torch tensor of non-covalent edges
        edge_attr_cov = torch tensor of non-covalent edge attributes
'''
# def get_edges_ncov(protein_chain, radius_ncov):

#     G_ncov = nx.Graph()

#     pos = protein_chain.GetConformers()[0].GetPositions()
#     dist_matrix = sp.spatial.distance_matrix(pos, pos)

#     node_idx = np.where((dist_matrix <= radius_ncov))
#     for i, j in zip(node_idx[0], node_idx[1]):
#         i = int(i)
#         j = int(j)

#         # omit covalent edges (to prevent double-counting of edges)
#         if (not protein_chain.GetBondBetweenAtoms(i, j)) and (i != j):
#             G_ncov.add_edge(i, j, type=1, dist=dist_matrix[i, j])
    
#     G_ncov = G_ncov.to_directed()

#     edge_index_ncov = torch.stack([torch.LongTensor((u, v)) for u, v in G_ncov.edges(data=False)]).T
#     edge_attr_ncov = torch.stack([torch.FloatTensor((a['type'], a['dist'])) for _, _, a in G_ncov.edges(data=True)])

#     return edge_index_ncov, edge_attr_ncov


'''
make_graph:
    about:
        creates .pyg torch graph Data object for single protein_path with atom nodes with relevant features, covalent edges, and non-covalent edges
    params:
        protein_path = file path to labeled protein or pocket .pdbs
        labels_path = file path to atom labels (b-factors) .npy
        graph_path = file path to torch graph Data object .pyg
        working_dir = working directory
        radius_ncov = cutoff radius for determining non-covalent edges around an atom in Angstroms
    returns: none
'''
def make_graph(protein_path, labels_path, graph_path, working_dir, radius_ncov):

    protein_chain = Chem.rdmolfiles.MolFromPDBFile(os.path.join(working_dir, protein_path), sanitize=True, removeHs=True, proximityBonding=False)
    labels = np.load(os.path.join(working_dir, labels_path))
    
    # get nodes and node features (x), edges and edge features (edge_index_cov, edge_index_ncov, edge_attr_cov, edge_attr_ncov)
    # and edge attributes (edge_attr), node labels (y), and node coordinates (pos)
    x = get_nodes(protein_chain)
    edge_index, edge_attr = get_edges(protein_chain, radius_ncov)
    # edge_index_cov, edge_attr_cov = get_edges_cov(protein_chain)
    # edge_index_ncov, edge_attr_ncov = get_edges_ncov(protein_chain, radius_ncov)
    y = torch.FloatTensor(labels)
    pos = torch.FloatTensor(protein_chain.GetConformers()[0].GetPositions())

    # cast graph components
    x = x.type(torch.FloatTensor)
    edge_index = edge_index.type(torch.LongTensor)
    edge_attr = edge_attr.type(torch.FloatTensor)
    # edge_index_cov = edge_index_cov.type(torch.LongTensor)
    # edge_index_ncov = edge_index_ncov.type(torch.LongTensor)
    # edge_attr_cov = edge_attr_cov.type(torch.FloatTensor)
    # edge_attr_ncov = edge_attr_ncov.type(torch.FloatTensor)
    y = y.type(torch.LongTensor)
    pos = pos.type(torch.FloatTensor)

    # save graph and log shape of graph components
    num_nodes = int(x.size()[0])
    num_node_features = int(x.size()[1])
    num_edges = int(edge_attr.size()[0])
    num_edge_features = int(edge_attr.size()[1])
    # num_edges_cov = int(edge_attr_cov.size()[0])
    # num_edges_ncov = int(edge_attr_ncov.size()[0])
    # num_edge_features_cov = int(edge_attr_cov.size()[1])
    # num_edge_features_ncov = int(edge_attr_ncov.size()[1])
    num_dimensions = int(pos.size()[1])

    graph = pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos,
                          num_nodes=num_nodes, num_node_features=num_node_features,
                          num_edges=num_edges, num_edge_features=num_edge_features,
                          num_dimensions=num_dimensions,
                          graph_path=graph_path, protein_path=protein_path, labels_path=labels_path, radius_ncov=radius_ncov)

    # graph = pyg.data.Data(x=x, edge_index_cov=edge_index_cov, edge_index_ncov=edge_index_ncov,
    #                       edge_attr_cov=edge_attr_cov, edge_attr_ncov=edge_attr_ncov, y=y, pos=pos,
    #                       num_nodes=num_nodes, num_node_features=num_node_features,
    #                       num_edges_cov=num_edges_cov, num_edges_ncov=num_edges_ncov,
    #                       num_edge_features_cov=num_edge_features_cov, num_edge_features_ncov=num_edge_features_ncov,
    #                       num_dimensions=num_dimensions,
    #                       graph_path=graph_path, protein_path=protein_path, labels_path=labels_path, radius_ncov=radius_ncov)

    torch.save(graph, os.path.join(working_dir, graph_path))

    logger = Logger('dataset.log')
    logger.info(f"[{graph_path}]: dim = {graph}")


'''
split_mask:
    about:
        creates training, validation, and testing masks from a list of protein lengths (num_nodes) and split ratio
    params:
        list_num_nodes = list of num_nodes in each protein
        split = training, validation, testing split ratio
    returns:
        train_mask = training mask
        valid_mask = validation mask
        test_mask = test mask
        list_mask_type = list of mask type applied to each protein
'''
# def split_mask(list_num_nodes, split):

#     num_subgraph_train = round(split[0]*len(list_num_nodes))
#     num_subgraph_valid = round(split[1]*len(list_num_nodes))
#     num_subgraph_test = round(split[2]*len(list_num_nodes))

#     subgraph_train = 0*np.ones((num_subgraph_train), dtype='long')
#     subgraph_valid = 1*np.ones((num_subgraph_valid), dtype='long')
#     subgraph_test = 2*np.ones((num_subgraph_test), dtype='long')

#     subgraph_mask_type = np.concatenate((subgraph_train, subgraph_valid, subgraph_test), axis=0, dtype='long')
#     np.random.shuffle(subgraph_mask_type)
#     subgraph_mask_type = list(subgraph_mask_type)

#     mask = np.array([], dtype='int')
#     for num_nodes, mask_type in zip(list_num_nodes, subgraph_mask_type):
#         mask = np.concatenate((mask, mask_type*np.ones((num_nodes), dtype='int'))) 

#     train_mask = mask==0
#     valid_mask = mask==1
#     test_mask = mask==2
#     list_mask_type = subgraph_mask_type

#     return train_mask, valid_mask, test_mask, list_mask_type


'''
combine_graphs:
    about:
        creates collated .pyg torch graph Data object for set of graph_paths with atom nodes with relevant features, covalent edges, and non-covalent edges
    params:
        graph_path_list = list of graph_paths to concatenate
        combined_graph_path = file path to combined torch graph Data object .pyg
        working_dir = working directory
        split = training, validation, testing split ratio
    returns:
'''
# def combine_graphs(graph_path_list, combined_graph_path, working_dir, split=[0.6, 0.2, 0.2]):
    
#     # initialize graph components
#     x = torch.empty((0,39), dtype=torch.float)
#     edge_index = torch.empty((2,0), dtype=torch.long)
#     edge_attr = torch.empty((0,2), dtype=torch.float)
#     y = torch.empty((0), dtype=torch.long)
#     pos = torch.empty((0,3), dtype=torch.float)

#     list_num_nodes = []
#     list_graph_path = []
#     list_protein_path = []
#     list_labels_path = []

#     # save start_idx for next protein
#     idx_node = 0

#     for graph_path in graph_path_list:

#         graph = torch.load(os.path.join(working_dir, graph_path))

#         x = torch.cat((x, graph.x), dim=0)

#         # shift indices of edge_index_cov and edge_index_ncov by start_idx
#         edge_index_cov_shifted = graph.edge_index_cov + idx_node
#         edge_index_ncov_shifted = graph.edge_index_ncov + idx_node
#         edge_index = torch.cat((edge_index, edge_index_cov_shifted, edge_index_ncov_shifted), dim=1)

#         edge_attr = torch.cat((edge_attr, graph.edge_attr_cov, graph.edge_attr_ncov), dim=0)
#         y = torch.cat((y, graph.y), dim=0)
#         pos = torch.cat((pos, graph.pos), dim=0)

#         list_num_nodes.append(graph.num_nodes)
#         list_graph_path.append(graph.graph_path)
#         list_protein_path.append(graph.protein_path)
#         list_labels_path.append(graph.labels_path)

#         idx_node += graph.num_nodes
    
#     # create training, validation, and test masks
#     train_mask, valid_mask, test_mask, list_mask_type = split_mask(list_num_nodes, split)

#     # save graph and log shape of graph components
#     x = x.type(torch.FloatTensor)
#     edge_index = edge_index.type(torch.LongTensor)
#     edge_attr = edge_attr.type(torch.FloatTensor)
#     y = y.type(torch.LongTensor)
#     pos = pos.type(torch.FloatTensor)

#     train_mask = torch.tensor(train_mask, dtype=torch.bool)
#     valid_mask = torch.tensor(valid_mask, dtype = torch.bool)
#     test_mask = torch.tensor(test_mask, dtype = torch.bool)
#     list_mask_type = torch.tensor(list_mask_type, dtype = torch.long)

#     num_nodes = int(x.size()[0])
#     num_node_features = int(x.size()[1])
#     num_edges = int(edge_index.size()[1])
#     num_edge_features = int(edge_attr.size()[1])
#     num_dimensions = int(pos.size()[1])

#     combined_graph = pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos,
#                                    train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask,
#                                    num_nodes=num_nodes, num_node_features=num_node_features,
#                                    num_edges=num_edges, num_edge_features=num_edge_features,
#                                    num_dimensions=num_dimensions, split=split,
#                                    list_mask_type=list_mask_type, list_num_nodes=list_num_nodes,
#                                    list_graph_path=list_graph_path, list_protein_path=list_protein_path, list_labels_path=list_labels_path)

#     torch.save(combined_graph, os.path.join(working_dir, combined_graph_path))

#     logger = Logger('dataset.log')
#     logger.info(f"[{combined_graph_path}]: dim = {combined_graph}")


'''
make_dataset:
    about:
        creates .pyg torch graph Data object dataset
    params:
        data_path = file path to dataframe of all processed data paths and parameters as .csv
        data_dir = file path to data folder
        working_dir = working directory
        data_type = string identifier for input data
        radius_ncov = cutoff radius for determining non-covalent edges around an atom in Angstroms
        split = training, validation, testing split ratio
        num_process = num_workers for multiprocessing
    returns: none
'''
def make_dataset(data_path, data_dir, working_dir, data_type='data', radius_ncov=10, split=[0.6, 0.2, 0.2], num_process=4):

    data_df = pd.read_csv(os.path.join(working_dir, data_path))

    # pull paths for labeled protein/pocket .pdb and label .npy from data_df dataframe
    protein_path_list = []
    labels_path_list = []
    graph_path_list = []
    working_dir_list = []
    radius_ncov_list = []

    for _, row in data_df.iterrows():
        protein_path_list.append(row['protein_path'])
        labels_path_list.append(row['labels_path'])
        graph_path_list.append(os.path.join(row['complex_dir'], f"{row['pdb_chain_id']}_graph.pyg"))
        working_dir_list.append(os.getcwd())
        radius_ncov_list.append(radius_ncov)

    # log parameter information
    logger = Logger('dataset.log')

    logger.critical("DATASET PARAMETERS")
    logger.info(f"data_path = {data_path}")
    logger.info(f"data_dir = {data_dir}")
    logger.info(f"data_type = {data_type}")
    logger.info(f"radius_ncov = {radius_ncov}")
    # logger.info(f"split = {split}")
    logger.info(f"LEN data_df = {len(data_df)}")

    logger.critical("DATASET START")

    # creates .pyg torch graph Data objects
    logger.critical(f"INDIVIDUAL GRAPHS")
    pool = multiprocessing.Pool(num_process)
    pool.starmap(make_graph, zip(protein_path_list, labels_path_list, graph_path_list, working_dir_list, radius_ncov_list))
    pool.close()
    pool.join()

    # logger.critical(f"COMBINED GRAPH")
    # split_str = '-'.join(str(format(s*100, '.0f')).zfill(2) for s in split)
    # combined_graph_path = os.path.join(data_dir, f"graph_TYPE-{data_type}_LEN-{len(data_df)}_SPLIT-{split_str}.pyg")
    # combine_graphs(graph_path_list, combined_graph_path, working_dir, split)

    logger.critical("DATASET END")

    return graph_path_list

'''
main:
    about:
        runs dataset generation pipeline
    params:
        data_dir = file path to data folder
        working_dir = working directory
        data_path = file path to dataframe of all processed data paths and parameters as .csv
        data_type = string identifier for input data
        radius_ncov = cutoff radius for determining non-covalent edges around an atom in Angstroms
        split = training, validation, testing split ratio
        num_process = num_workers for multiprocessing
    returns: none
'''
if __name__ == '__main__':

    data_dir = "data/processed"
    working_dir = os.getcwd()

    data_path = "data/processed/data_paths.csv"

    make_dataset(data_path, data_dir, working_dir, data_type='data', radius_ncov=10, split=[0.8, 0.0, 0.2], num_process=4)

