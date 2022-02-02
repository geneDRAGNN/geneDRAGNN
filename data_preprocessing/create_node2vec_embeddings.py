import pandas as pd
from pecanpy import node2vec

def Runnode2vec(filepath):
    n2v = node2vec.SparseOTF(p=1, q=1, workers=4, verbose=True)

    edge_list = n2v.read_edg(filepath, weighted=False, directed=False)
    emd = n2v.embed(dim=128, num_walks=10, walk_length=80, window_size=10, epochs=10)

    n2v_emd = pd.DataFrame(emd, n2v.IDlst)

    n2v_emd.columns = ['network_' + str(col) for col in n2v_emd.columns]

    return n2v_emd.reset_index().rename(columns={"index":"ensembl"})