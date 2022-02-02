import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric

# define GNN architecture
class GNNModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        hidden_dense,
        GNN_conv_layer=GCNConv,
        dropout_rate=0.1,
        **kwargs,
    ):
        """
        Args:
            in_channels (int): Dimension of input features
            hidden_channels (List[int]): Dimension of hidden features
            out_channels (int): Dimension of the output.
            hidden_dense (int): number of units in hidden dense layer following convolutions.
            GNN_conv_layer: Class of the graph convolutional layer to use.
            dropout_rate (float): Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()

        self.convs = []
        self.convs.append(
            GNN_conv_layer(
                in_channels=in_channels, out_channels=hidden_channels[0], **kwargs
            )
        )  # first GNN Conv layer

        for c1, c2 in zip(hidden_channels[:-1], hidden_channels[1:]):  # middle layers
            self.convs.append(GNN_conv_layer(in_channels=c1, out_channels=c2, **kwargs))

        self.convs = torch.nn.ModuleList(self.convs)

        self.dense1 = torch.nn.Linear(hidden_channels[-1], hidden_dense)
        self.dense_out = torch.nn.Linear(hidden_dense, num_classes)

        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: node features
            edge_index: edge list
        """

        for i, conv in enumerate(self.convs):
            if edge_attr is None:
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            x = x.relu()
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.dense1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.dense_out(x)

        return x


# define Pytorch Lightning model
class LitGNN(pl.LightningModule):
    def __init__(self, model_name, model=None, model_type="edge", **model_kwargs):
        super().__init__()

        # Saving hyperparameters
        self.save_hyperparameters()

        self.model_name = model_name

        self.model_type = model_type
        if self.model_type not in ["edge", "edge_attr", "baseline"]:
            raise TypeError(
                "Invalid `model_type`. Must be one of ['edge', 'edge_attr', 'baseline']"
            )

        # create model using GNNModel if one isn't given
        if model is None:
            self.model = GNNModel(**model_kwargs)
        else:
            self.model = model(**model_kwargs)

        # define the loss function
        self.loss_module = torch.nn.CrossEntropyLoss()

        # give example input
        self.example_input_array = data

    def forward(self, data, mode="train"):
        if self.model_type == "edge_attr":
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            x = self.model(x, edge_index, edge_attr=edge_attr)
        elif self.model_type == "edge":
            x, edge_index = data.x, data.edge_index
            x = self.model(x, edge_index)
        else:
            x = data.x
            x = self.model(x)

        # Only calculate the loss and acc on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            raise ValueError(f"Unknown forward mode: {mode}")

        # TODO: add other metrics like recall, precision, f1, etc...
        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()

        return x, loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters()
        )  # SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, loss, acc = self.forward(batch, mode="train")
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss, acc = self.forward(batch, mode="val")
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return logits

    def validation_epoch_end(self, validation_step_outputs):

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        if self.logger:
            self.logger.experiment.log(
                {
                    "val_logits": wandb.Histogram(flattened_logits.to("cpu")),
                    "global_step": self.global_step,
                }
            )

    def test_step(self, batch, batch_idx):
        x, _, acc = self.forward(batch, mode="test")
        self.log(
            "test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )


def create_SGConv_GNN(project_name, num_features, num_classes):
    """creates SGConv GNN model"""
    GNN_conv_layer = torch_geometric.nn.SGConv
    model = LitGNN(
        project_name,
        model=None,
        GNN_conv_layer=GNN_conv_layer,
        in_channels=num_features,
        hidden_channels=[128, 256, 256, 128],
        out_channels=num_classes,
        hidden_dense=128,
        model_type="edge",
    )
    return model


def create_GraphSAGE_GNN(project_name, num_features, num_classes):
    """creates GraphSAGE GNN model"""

    model_ = torch_geometric.nn.models.GraphSAGE  # base model
    hidden_channels = 256
    num_layers = 3
    dropout = 0.2

    model = LitGNN(
        project_name,
        model=model_,
        jk="max",
        in_channels=num_features,
        hidden_channels=hidden_channels,
        dropout=dropout,
        num_layers=num_layers,
        out_channels=num_classes,
        model_type="edge",
    )

    return model


def create_TAG_GNN(project_name, num_features, num_classes):
    '''creates TAG GNN model'''

    GNN_conv_layer = torch_geometric.nn.TAGConv

    model = LitGNN(
        project_name,
        model=None,
        GNN_conv_layer=GNN_conv_layer,
        K=3,
        in_channels=num_features,
        hidden_channels=[128, 256, 128],
        out_channels=num_classes,
        hidden_dense=256,
        model_type="edge",
    )

    return model


def create_clusterGCN_GNN(project_name, num_features, num_classes):
    '''creates clusterGCN GNN model'''

    GNN_conv_layer = torch_geometric.nn.ClusterGCNConv

    model = LitGNN(
        project_name,
        model=None,
        GNN_conv_layer=GNN_conv_layer,
        in_channels=num_features,
        hidden_channels=[128, 256, 256, 128],
        out_channels=num_classes,
        hidden_dense=128,
    )

    return model

