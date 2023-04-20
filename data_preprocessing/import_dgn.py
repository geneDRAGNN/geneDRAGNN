import pandas as pd

def ImportDGN():
    dgn = pd.read_csv("../data/raw_data/gda_disease_summary_luad.csv")
    dgn_dict = pd.read_csv("../data/raw_data/gda_dictionary.csv", index_col=None)

    score_threshold = 0.02
    ei_threshold = 0.7

    dgn = dgn[['Gene', 'EI_gda', 'Score_gda']]
    dgn = dgn.loc[dgn['Score_gda'] >= score_threshold]
    dgn = dgn.loc[dgn['EI_gda'] > ei_threshold]
    dgn.rename({'Score_gda':'gda_score'}, axis=1, inplace=True)
    dgn = dgn.merge(dgn_dict, on="Gene").drop(['Gene'], axis=1)
    dgn['gda_score'] = 1

    return dgn[['ensembl', 'gda_score']]