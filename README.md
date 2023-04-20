# geneDRAGNN - gene Disease Prioritization using Graph Neural Networks

Authors: Awni Altabaa, David Huang, Ciaran Byles-Ho, Hani Khatib, Fabian Sosa-Miranda, Ting Hu

Link to the paper: https://ieeexplore.ieee.org/document/9863043

## Abstract
Many human diseases exhibit a complex genetic etiology impacted by various genes and proteins in a large network of interactions. The process of evaluating gene-disease associations through in-vivo experiments is both time-consuming and expensive. Thus, network-based computational methods capable of modeling the complex interplay between molecular components can lead to more targeted evaluation. In this paper, we propose and evaluate geneDRAGNN: a general data processing and machine learning methodology for exploiting information about gene-gene interaction networks for predicting gene-disease association. We demonstrate that information derived from the gene-gene interaction network can significantly improve the performance of gene-disease association prediction models. We apply this methodology to lung adenocarcinoma, a histological subtype of lung cancer. We identify new potential gene-disease associations and provide supportive evidence for the association through gene-set enrichment and literature based analysis.


## Supplementary Material
The supplementary material can be found [here](geneDRAGNN_Supplementary.pdf).

## Repo Structure
- `data`: This directory contains information about the datasets used in the paper.
- `data_preprocessing`: This directory contains code used to preprocess the data.
- `gene_enrichment_analysis`: This directory contains the results of the gene enrichment analysis.
- `literature_review`: This directory contains the results of the literature review performed on our model's top-ranked genes.
- `models`: This directory contains code related to our models, including: architecture definitions, training, and inference.
