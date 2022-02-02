import pandas as pd

def ImportGDC():
    gdc = pd.read_csv("../data/raw_data/gdc_luad_genes.csv")

    featureNameColumns = ['# SSM Affected Cases in Cohort', '# CNV Gain', '# CNV Loss']
    for i in featureNameColumns:
        gdc[['1', '2', '3', '4']] = gdc[i].replace({',':''}, regex=True).str.split(' ', 3, expand=True)
        gdc[i] = gdc['1'].astype(float)/gdc['3'].astype(float)
        gdc = gdc.drop(['1', '2', '3', '4'], axis=1)

    gdc.drop(['Symbol', 'Name', 'Cytoband', 'Type', 'Annotations', 'Survival'], axis=1, inplace=True)

    gdc[['1', '2', '3']] = gdc['# SSM Affected Cases Across the GDC'].replace({',':''}, regex=True).str.split(' ', 2, expand=True)
    gdc['# SSM Affected Cases Across the GDC'] = gdc['1'].astype(float)/gdc['3'].astype(float)
    gdc = gdc.drop(['1', '2', '3'], axis=1)

    gdc = gdc.rename({'# SSM Affected Cases in Cohort': 'nih_ssm_in_cohort', '# SSM Affected Cases Across the GDC':'nih_ssm_across_gdc',
        '# CNV Gain':'nih_cnv_gain', '# CNV Loss':'nih_cnv_loss', 'Gene ID':'ensembl', '# Mutations':'nih_tot_mutations'}, axis=1)

    return gdc