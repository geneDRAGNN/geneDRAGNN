# Data Processing
The scripts and functions required for processing the required datasets into the required format for the models.

## Main Scripts

### main_data_pipeline.ipynb
A notebook that conducts the full data processing from start to finish by importing the features, edges and labels separately and providing the necessary operations to make the final datasets.

## Usable Functions
Note: Functions were made for the use of the November 2021 data and may not function with updated datasets. Please email ciaran.bylesho@gmail.com to receive the required datasets to rerun the experiments.

### create_node2vec_embeddings.py
Imports the pecanpy library and applys an optimized node2vec to the target edgelist to create embeddings.

### import_dgn.py
Imports the Disease Gene Network data and processes it it provide gene disease association scores and evidence index scores.

### import_gdc.py
Imports the National Institute of Health: Genomic Data Commons data and processes it it provide mutation features.

### import_hpa.py
Imports the Human Protein Atlas data and processes it it provide genetic based features.

### import_string.py
Imports the STRING data and processes it it provide the edge list and edge list features.
