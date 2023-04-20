# Models

This subdirectory contains code related to building and training gene-disease association prediction models.

- `baseline_models.py` contains code for training baseline node-only models.
- `train_baseline_model.ipynb` trains those baseline models.
- `models.py` contains the architecture definition of the graph neural network models which were examined in the paper.
` model_utils.py` contains miscellaneous utility functions used while training/evaluating models.
- `data_utils.py` contains utility functions related to loading data for training and inference.
- `train_baseline_models.ipynb` trains the GNN models in `models.py`
- the `model_checkpoints` directory contains trained model checkpoints.
- `run_gnn_model.ipynb` shows how to build a model from `models.py`, load a pretrained checkpoint, and run it on the dataset.

In our experiments, we used a common architecture for our GNNs, and iterated with different types of graph convolution layers. Each GNN takes as input the LS-GIN (represented as an edge list) and the gene ontology features of each node. Information is then propagated from each gene to its neighbours in the LS-GIN through a series of learnable graph convolution layers. With each successive convolution operation, information is propagated one degree further. At the end of these graph convolutions, each gene is left with a vector embedding representing the relevant aspects of its gene-ontology features and its position within the LS-GIN. These embeddings are finally passed through a dense feed-forward neural network which produces the gene-disease association prediction.

<br>
<center><a href="https://ibb.co/2jCmM1w"><img src="https://i.ibb.co/4jX9Snc/GNN-Architecture-horizontal-v3.jpg" alt="GNN-Architecture-horizontal-v3" border="0"></a></center>
<br>

The GNN architectures we tested include Simple Graph Convolutional Networks (SGC), Topology Adaptive Graph Convolutional Networks (TAGCN), Cluster-GCN, and GraphSAGE. These all fall under the "convolutional" sub-class of GNNs. This is the simplest class graph convolution operations. It was necessary to restrict our exploration to "convolutional" GNNs due to computational limitations related to the large size of the LS-GIN. The choice of hyperparameters is motivated by the network properties of the LS-GIN. In particular, the diameter of LS-GIN motivates a shallow network of 3 or 4 graph convolution layers.

In general, models trained on LS-GIN embeddings consistently outperformed the baseline models trained exclusively on gene-ontology features. Graph-based models achieved a maximum accuracy of 78%, a maximum precision of 80%, and a maximum recall of 75%. The model which achieved the highest average accuracy was the Support Vector Machine trained on gene-ontology features and node2vec embeddings of LS-GIN with an average accuracy of 78.0% and a standard deviation of 2.2%. The highest performing GNNs achieved an average accuracy of roughly 75%. SGConv achieved an average accuracy of 74.3% with a standard deviation of 2.7%, and an average precision of 74.6%. The SGConv model had the highest positive recall of all GNN models, averaging 75.0%. The topology adaptive graph convolutional network achieved a similar average accuracy of 74.9%, an average precision of 77.8%, and an average recall of 70.6% (though this model was only evaluated on 10 trials). Cluster-GCN and GraphSAGE achieved a lower performance with an average accuracy of 72.6% and 71.4%, respectively.

<br>

<div align="center">

| Model              | Features Used                                    | Accuracy (avg) | Accuracy (std) | Recall (avg) | Recall (std) | Precision (avg) | Precision (std) | F1 (avg) | F1 (std) | # of Trials |
|--------------------|--------------------------------------------------|----------------|----------------|--------------|--------------|-----------------|-----------------|----------|----------|-------------|
| Baseline Models    |                                                  |                |                |              |              |                 |                 |          |          |             |
| RF                 | node features                                    | 0.707          | 0.027          | 0.728        | 0.037        | 0.700           | 0.030           | 0.707    | 0.027    | 100         |
| MLP                | node features                                    | 0.699          | 0.025          | 0.668        | 0.029        | 0.714           | 0.035           | 0.699    | 0.025    | 100         |
| SVM                | node features                                    | 0.693          | 0.022          | 0.506        | 0.040        | 0.809           | 0.039           | 0.681    | 0.024    | 100         |
| KNN                | node features                                    | 0.645          | 0.024          | 0.451        | 0.042        | 0.737           | 0.040           | 0.630    | 0.026    | 100         |
| Graph-Based Models |                                                  |                |                |              |              |                 |                 |          |          |             |
| RF                 | node2vec features                                | 0.765          | 0.022          | 0.666        | 0.039        | 0.831           | 0.030           | 0.762    | 0.022    | 100         |
| RF                 | node features, node2vec features                 | 0.766          | 0.022          | 0.705        | 0.037        | 0.803           | 0.027           | 0.765    | 0.022    | 100         |
| MLP                | node2vec features                                | 0.747          | 0.025          | 0.751        | 0.039        | 0.746           | 0.029           | 0.747    | 0.025    | 100         |
| MLP                | node features, node2vec features                 | 0.744          | 0.026          | 0.735        | 0.038        | 0.749           | 0.031           | 0.744    | 0.026    | 100         |
| SVM                | node2vec features                                | 0.780          | 0.022          | 0.758        | 0.036        | 0.794           | 0.028           | 0.780    | 0.022    | 100         |
| SVM                | node features, node2vec features                 | 0.705          | 0.021          | 0.530        | 0.036        | 0.815           | 0.033           | 0.695    | 0.022    | 100         |
| KNN                | node2vec features                                | 0.564          | 0.017          | 0.949        | 0.023        | 0.537           | 0.010           | 0.488    | 0.029    | 100         |
| KNN                | node features, node2vec features                 | 0.645          | 0.026          | 0.752        | 0.041        | 0.620           | 0.025           | 0.640    | 0.027    | 100         |
| SGCN               | node features, LS-GIN network, edge features     | 0.750          | 0.019          | 0.774        | 0.080        | 0.743           | 0.034           | 0.749    | 0.019    | 11          |
| SGCN               | node features, LS-GIN network                    | 0.743          | 0.028          | 0.750        | 0.084        | 0.746           | 0.050           | 0.741    | 0.028    | 100         |
| GraphSAGE          | node features, LS-GIN network                    | 0.714          | 0.015          | 0.674        | 0.040        | 0.733           | 0.020           | 0.713    | 0.016    | 10          |
| GraphSAGE          | node features, LS-GIN network, node2vec features | 0.745          | 0.020          | 0.721        | 0.047        | 0.759           | 0.027           | 0.745    | 0.021    | 16          |
| TAGCN              | node features, LS-GIN network                    | 0.749          | 0.024          | 0.706        | 0.067        | 0.778           | 0.049           | 0.747    | 0.024    | 10          |
| TAGCN              | node features, LS-GIN network, node2vec features | 0.741          | 0.035          | 0.726        | 0.047        | 0.750           | 0.042           | 0.741    | 0.035    | 10          |
| Cluster-GCN        | node features, LS-GIN network, edge features     | 0.726          | 0.020          | 0.671        | 0.068        | 0.757           | 0.031           | 0.724    | 0.021    | 11          |

</dev>