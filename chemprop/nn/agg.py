from abc import abstractmethod

import torch
from torch import Tensor, nn

from chemprop.nn.hparams import HasHParams
from chemprop.utils import ClassRegistry
from chemprop.nn.set_transformer_models import SetTransformer

__all__ = [
    "Aggregation",
    "AggregationRegistry",
    "MeanAggregation",
    "SumAggregation",
    "NormAggregation",
    "AttentiveAggregation",
    "SetTransformerAggregation",
]


class Aggregation(nn.Module, HasHParams):
    """An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
    a batch of graph-level representations

    .. note::
        this class is abstract and cannot be instantiated.

    See also
    --------
    :class:`~chemprop.v2.models.modules.agg.MeanAggregation`
    :class:`~chemprop.v2.models.modules.agg.SumAggregation`
    :class:`~chemprop.v2.models.modules.agg.NormAggregation`
    """

    def __init__(self, dim: int = 0, *args, **kwargs):
        super().__init__()

        self.dim = dim
        self.hparams = {"dim": dim, "cls": self.__class__}

    @abstractmethod
    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        """Aggregate the graph-level representations of a batch of graphs into their respective
        global representations

        NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
        a zero vector of length `d` in the final output.

        Parameters
        ----------
        H : Tensor
            a tensor of shape ``V x d`` containing the batched node-level representations of ``b``
            graphs
        batch : Tensor
            a tensor of shape ``V`` containing the index of the graph a given vertex corresponds to

        Returns
        -------
        Tensor
            a tensor of shape ``b x d`` containing the graph-level representations
        """


AggregationRegistry = ClassRegistry[Aggregation]()


@AggregationRegistry.register("mean")
class MeanAggregation(Aggregation):
    r"""Average the graph-level representation:

    .. math::
        \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v
    """

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        dim_size = batch.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, H, reduce="mean", include_self=False
        )


@AggregationRegistry.register("sum")
class SumAggregation(Aggregation):
    r"""Sum the graph-level representation:

    .. math::
        \mathbf h = \sum_{v \in V} \mathbf h_v

    """

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        dim_size = batch.max().int() + 1
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, H, reduce="sum", include_self=False
        )


@AggregationRegistry.register("norm")
class NormAggregation(SumAggregation):
    r"""Sum the graph-level representation and divide by a normalization constant:

    .. math::
        \mathbf h = \frac{1}{c} \sum_{v \in V} \mathbf h_v
    """

    def __init__(self, dim: int = 0, *args, norm: float = 100.0, **kwargs):
        super().__init__(dim, **kwargs)

        self.norm = norm
        self.hparams["norm"] = norm

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        return super().forward(H, batch) / self.norm


class AttentiveAggregation(Aggregation):
    def __init__(self, dim: int = 0, *args, output_size: int, **kwargs):
        super().__init__(dim, *args, **kwargs)

        self.W = nn.Linear(output_size, 1)

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        dim_size = batch.max().int() + 1
        attention_logits = self.W(H).exp()
        Z = torch.zeros(dim_size, 1, dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, batch.unsqueeze(1), attention_logits, reduce="sum", include_self=False
        )
        alphas = attention_logits / Z[batch]
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        return torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            self.dim, index_torch, alphas * H, reduce="sum", include_self=False
        )

# class SetTransformerAggregation(Aggregation):
#     SetTransformer(dim_input=emb_dim*n_heads, num_outputs=1, dim_output=1).to(device)

class SetTransformerAggregation(Aggregation):
    def __init__(self, dim: int = 0, *args, output_size: int, num_heads: int = 4, dim_hidden: int = 128, **kwargs):
        super().__init__(dim, *args, **kwargs)
        
        self.set_transformer = SetTransformer(
            dim_input=output_size,
            num_outputs=1,  # We want one output per set
            dim_output=output_size,  # Maintain same dimension as input
            dim_hidden=dim_hidden,
            num_heads=num_heads
        )
        
        # self.hparams.update({
        #     "input_dim": input_dim,
        #     "num_heads": num_heads,
        #     "dim_hidden": dim_hidden
        # })

    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        # Get dimensions
        dim_size = batch.max().int() + 1  # Number of graphs in batch
        
        # # Create a padded tensor where each row is a graph's node features
        # max_nodes = torch.bincount(batch).max()

        # padded_H = torch.zeros(dim_size, max_nodes, H.shape[1], dtype=H.dtype, device=H.device)
        # for i_graph in range(dim_size):
        #     node_mask = batch == i_graph
        #     graph = H[node_mask, :]
        #     padded_H[i_graph, :graph.size(0)] = graph

        # return self.set_transformer(padded_H).squeeze()
        

        agg_graph = torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device)
        max_graph_size = 0
        for i_graph in range(batch.unique().size(0)):
            node_mask = batch == i_graph
            graph = H[node_mask, :].unsqueeze(0)
            agg_graph[i_graph, :] = self.set_transformer(graph)
            max_graph_size = max(max_graph_size, graph.size(1))
        return agg_graph