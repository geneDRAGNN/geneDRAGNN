from email.mime import base
import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np


def read_data(node_filepath="../data/final_data/node_node2vec_data.csv",
              label_filepath="../data/final_data/training_labels_trials.csv",
              edgelist_path="../data/final_data/ls-fgin_edge_list.edg",
              feats_type='nodeonly'):
    """
    reads data (node features, edge list, labels) from local files.

    Args:
        base_path (str): directory in which data is stored.
        graph_used (str, optional): type of graph. 'know' or 'phys'. Defaults to 'know'.
        feats_type (str, optional): type of features. 'nodeonly' or 'node+net'. Defaults to 'nodeonly'.
        label_thres (str, optional): gda score threshold for labels. Defaults to '0,02'.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: node_dataset, edge_list, labels
    """


    # LOAD NODE DATA
    node_dataset = pd.read_csv(node_filepath, index_col=0)
    if feats_type == 'nodeonly':
        node_dataset = node_dataset[node_dataset.columns.drop(list(node_dataset.filter(regex='network')))]

    elif feats_type == 'networkonly':
        node_dataset = node_dataset[node_dataset.columns.drop(list(node_dataset.filter(regex='nih')))]
        node_dataset = node_dataset[node_dataset.columns.drop(list(node_dataset.filter(regex='hpa')))]
    

    # assert that all genes in dataset are unique
    assert(node_dataset.index.duplicated().sum() == 0)

    # create mapping to 0-based index map
    genes = node_dataset.index.to_numpy()
    gene_id_dict = {ensembl: idx for idx, ensembl in enumerate(genes)}

    # map ID's in node dataset
    myID = node_dataset.index.map(gene_id_dict).rename('myID')
    node_dataset.insert(loc=0, column='myID', value=myID)
    node_dataset = node_dataset.reset_index().set_index('myID')
    node_dataset.drop(columns=['ensembl'], inplace=True)

    # LOAD LABELS
    labels = pd.read_csv(label_filepath, index_col=0)

    # drop label columns from node_dataset
    label_cols_drop = ['gda_score']
    labels.drop(columns=label_cols_drop, inplace=True)

    # map Ensmebl index to myID in labels dataset
    myID = labels.index.map(gene_id_dict).rename('myID')
    labels.insert(loc=0, column='myID', value=myID)
    labels = labels.reset_index().set_index('myID')
    labels.drop(columns=['ensembl'], inplace=True)

    # convert type
    num_label_trials = 100
    for label_col in [f'label_{i}' for i in range(num_label_trials)]:
        labels[label_col] = labels[label_col].astype('Int32')

    edge_list = pd.read_csv(edgelist_path, header=None, sep='\t')
    
    # map edge list
    edge_list.iloc[:, 0] = edge_list.iloc[:, 0].map(gene_id_dict)
    edge_list.iloc[:, 1] = edge_list.iloc[:, 1].map(gene_id_dict)

    # scale edge features appropriately (they take values in the range 0-1000)
    if len(edge_list.columns) > 2:
        edge_feat_cols = edge_list.columns[2:].to_numpy()
        edge_list[edge_feat_cols] /= 1000

    return node_dataset, edge_list, labels

from sklearn.model_selection import train_test_split
def get_train_val_test_masks(node_data_labeled, n_nodes, label_col, test_size=0.2, val_size=0.1):
    """generates a train-val-test split and returns masks.

    Args:
        node_data_labeled (DataFrame): labeled subset of node features dataframe.
        n_nodes (int): number of nodes in full dataset.
        label_col (str): name of label column.
        test_size (float, optional): size of test split. Defaults to 0.2.
        val_size (float, optional): size of val split. Defaults to 0.1.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: train_mask, val_mask, test_mask
    """

    X_myIDs = node_data_labeled.index.to_numpy() # myIDs for nodes with labels for training/testing
    labels = node_data_labeled[label_col].to_numpy() # for stratification

    test_size = test_size
    val_size = val_size * (1/(1-test_size))

    myIDs_train_val, myIDs_test = train_test_split(X_myIDs, test_size=test_size, shuffle=True, stratify=labels)

    labels_train_val = node_data_labeled.loc[myIDs_train_val][label_col].to_numpy()
    myIDs_train, myIDs_val = train_test_split(myIDs_train_val, test_size=val_size, shuffle=True, stratify=labels_train_val)

    # NOTE: train-val-test split is shuffled and stratified

    # create masks
    train_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[myIDs_train] = True
    train_mask = torch.Tensor(train_mask).type(torch.bool)

    val_mask = np.zeros(n_nodes, dtype=bool)
    val_mask[myIDs_val] = True
    val_mask = torch.Tensor(val_mask).type(torch.bool)

    test_mask = np.zeros(n_nodes, dtype=bool)
    test_mask[myIDs_test] = True
    test_mask = torch.Tensor(test_mask).type(torch.bool)

    return train_mask, val_mask, test_mask


def create_data(node_dataset, edge_list, labels, label_col, test_size=0.2, val_size=0.1):
    """creates training-ready data in pytorch_geometric format.

    Args:
        node_dataset (DataFrame): node dataset.
        edge_list (DataFrame): edge list dataframe.
        labels (DataFrame): labels dataframe with trials of labels.
        label_col (str): name of column in labels dataframe.
        test_size (float, optional): test split size. Defaults to 0.2.
        val_size (float, optional): val split size. Defaults to 0.1.

    Returns:
        torch_geometric.data.Data: Data object with node features, edge list,
            edge attributes, and train-val-test masks.
    """

    # get subset of node features features + labels
    node_labels = node_dataset.merge(labels[label_col], left_on='myID', right_on='myID')

    node_feat_cols = node_labels.columns[:-1]

    X = torch.Tensor(node_labels[node_feat_cols].to_numpy())#.type(torch.float64)

    y = node_labels[label_col].fillna(-1).astype('int') # fill NaN with -1 so that it can be converted to pytorch tensor
    y = torch.Tensor(y).type(torch.int64)

    # extract edges by index
    edge_index = torch.Tensor(edge_list.iloc[:, :2].to_numpy().T).type(torch.int64)

    # extract edge features
    edge_feat_cols = edge_list.columns[2:].to_numpy()
    edge_attr = torch.Tensor(edge_list[edge_feat_cols].to_numpy())

    # restrict to data with labels
    node_data_labeled = node_labels[node_labels[label_col].notna()]
    n_nodes = len(node_dataset) # total number of nodes
    train_mask, val_mask, test_mask = get_train_val_test_masks(node_data_labeled, n_nodes, label_col, test_size=test_size, val_size=val_size)

    data = Data(x=X, y=y, edge_index=edge_index, edge_attr=edge_attr)

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
