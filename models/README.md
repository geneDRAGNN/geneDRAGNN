##### projectx Queen's
# Final Results

#### Results table: The performance achieved by each model was tested. The baseline models use only node features–features about the genes themselves. The graph-based models use features about the gene-gene interaction network (either via node2vec embeddings or a GNN processing the graph directly). The metrics are averaged over multiple trials. Some models were evaluated on fewer than the full 100 trials due time and computational constraints. The more promising models were evaluated on the full 100 trials. The constraint for most GNNs was our computational memory, as our functional protein network was too large for our limited computing resources to handle.

<img width="1059" alt="Screen Shot 2022-02-02 at 3 15 26 AM" src="https://user-images.githubusercontent.com/85202161/152117149-c73e304a-faf9-4961-9ee1-53ced66d4cf3.png">

![image](https://user-images.githubusercontent.com/85202161/152122935-980b955f-e2cd-4834-8ac7-a4ab92e2c7de.png)

<img width="792" alt="Functional" src="https://user-images.githubusercontent.com/85202161/152126430-da9c1024-d058-4ce4-bb48-6aa548df897f.png">

<img width="797" alt="Functional Enrichment Analysis" src="https://user-images.githubusercontent.com/85202161/152126448-3631011b-375b-456b-91f2-b1eff0f9c684.png">

<img width="877" alt="Functional" src="https://user-images.githubusercontent.com/85202161/152126585-3c927b59-e7cd-4821-a48d-e08c48677baf.png">

<img width="719" alt="KEGG Pathway Enrichment Analysis" src="https://user-images.githubusercontent.com/85202161/152126599-0de0bcd4-1cab-49be-a04a-795d547d756c.png">


# Graph Neural Network Architectures:
The related code, visuals and datasets related to the Queen's University Project X research project in 2022. This visuals and descriptions of the GNNs in this section are retrieved from [this Graph Neural Net blog on Github](https://github.com/hanikhatib/graph_nets) 


## [Graph Convolutional Network (GCN)](https://dsgiitr.com/blogs/gcn/)
![gcn_architecture](https://user-images.githubusercontent.com/85202161/152111587-6dfda848-7d1c-4087-a858-a5da08cc844d.png)

### "GCNs draw on the idea of Convolution Neural Networks re-defining them for the non-euclidean data domain. They are convolutional, because filter parameters are typically shared over all locations in the graph unlike typical GNNs." [[1]](https://dsgiitr.com/blogs/gcn/)


## [Graph SAmple and aggreGatE (GraphSAGE)](https://dsgiitr.com/blogs/graphsage/)
![GraphSAGE_cover](https://user-images.githubusercontent.com/85202161/152111614-edef5e77-94d3-40de-821e-a46b6a6347b6.jpeg)

### "Previous approaches are transductive and don't naturally generalize to unseen nodes. GraphSAGE is an inductive framework leveraging node feature information to efficiently generate node embeddings." [[2]](https://dsgiitr.com/blogs/graphsage/).


## [ChebNet: CNN on Graphs with Fast Localized Spectral Filtering](https://dsgiitr.com/blogs/chebnet/)
![ChebNet_Cover](https://user-images.githubusercontent.com/85202161/152111710-503a436d-c054-45e4-84e7-0136ba14b05f.jpeg)

### "ChebNet is a formulation of CNNs in the context of spectral graph theory." [[3]](https://dsgiitr.com/blogs/chebnet/)


## [Graph Attention Netork (GAT)](https://dsgiitr.com/blogs/gat/)
![GAT_Cover](https://user-images.githubusercontent.com/85202161/152111692-cdfdb32e-184d-4d2e-aa9b-7069322ecb80.jpeg)

### "GAT is able to attend over their neighborhoods’ features, implicitly specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation or depending on knowing the graph structure upfront." [[4]](https://dsgiitr.com/blogs/gat/)


## [Simple Graph Convolutional Network (SGConv)](https://github.com/Tiiiger/SGC)
![SGConv](https://user-images.githubusercontent.com/85202161/152112089-bb6a3444-5ddc-4f2c-a4c8-2905315f01a2.jpeg)

### "SGC removes the nonlinearities and collapes the weight matrices in Graph Convolutional Networks (GCNs) and is essentially a linear model" [[5]](https://github.com/Tiiiger/SGC)


## [Topology Adaptive Graph Convolutional Network (TAGCN)](https://medium.com/@lavenderchiang/topology-adaptive-graph-cnn-8c4dffff858e)
![TAGCN](https://user-images.githubusercontent.com/85202161/152113038-1b62bf19-eb08-436b-9cdd-421af6e496b5.png)

### "The TAGCN not only inherits the properties of convolutions in CNN for grid-structured data, but it is also consistent with convolution as defined in graph signal processing. Since no approximation to the convolution is needed, TAGCN exhibits better performance than existing spectral CNNs on a number of data sets and is also computationally simpler than other recent methods." [[6]](https://arxiv.org/abs/1710.10370)
