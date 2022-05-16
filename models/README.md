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

| Model                  | Features Used                                              | Accuracy | Positive Recall | Positive Precision | F1-Score | # of Trials |
|------------------------|------------------------------------------------------------|------------------------------|-------------------------------------|----------------------------------------|------------------------------|----------------------------------|
| Random Forest          | Node features                                              | **0.707**               | **0.728**                      | 0.699                                  | **0.707**               | 100                              |
| MLP                    | Node features                                              | 0.699                        | 0.668                               | 0.714                                  | 0.699                        | 100                              |
| K-Nearest Neighbours   | Node features                                              | 0.645                        | 0.45                                | 0.737                                  | 0.630                        | 100                              |
| Support Vector Machine | Node features                                              | 0.693                        | 0.506                               | **0.809**                         | 0.681                        | 100                              |
| Random Forest          | Node Features, node2vec Network Features                   | **0.766**               | 0.705                               | **0.802**                         | 0.765                        | 100                              |
| K-Nearest Neighbours   | Node Features, node2vec Network Features                   | 0.645                        | 0.751                               | 0.620                                  | 0.640                        | 100                              |
| Support Vector Machine | node2vec Network Features                                  | 0.780                        | **0.759**                      | 0.794                                  | **0.780**               | 100                              |
| MLP                    | Node Features, node2vec Network Features                   | 0.744                        | 0.735                               | 0.749                                  | 0.744                        | 100                              |
| MLP                    | node2vec Network Features                                  | 0.731                        | 0.736                               | 0.730                                  | 0.730                        | 100                              |
| SGConv GNN             | Node features, Functional Graph                            | 0.743                        | 0.750                               | 0.746                                  | 0.741                        | 100                              |
| TAGCN                  | Node Features, Functional Graph                            | 0.749                        | 0.706                               | 0.778                                  | 0.747                        | 10                               |
| TAGCN                  | Node Features, node2vec Network Features, Functional Graph | 0.741                        | 0.726                               | 0.750                                  | 0.741                        | 10                               |
| Cluster-GCN            | Node Features, Functional Graph                            | 0.726                        | 0.671                               | 0.757                                  | 0.724                        | 11                               |
| GraphSAGE              | Node Features, Functional Graph                            | 0.714                        | 0.674                               | 0.733                                  | 0.713                        | 10                               |
</div>