from tqdm import tqdm
from functools import partial
from collections import defaultdict
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scGraphLLM.data import GraphTransformerDataset, send_to_gpu
from scGraphLLM.vocab import GeneVocab
from scGraphLLM.network import RegulatoryNetwork
from scGraphLLM.tokenizer import GraphTokenizer


class InferenceDataset(GraphTransformerDataset):
    """
    Dataset class for performing inference with a fixed regulatory network.

    Each cell's expression profile is tokenized using the provided tokenizer.
    This dataset is used to generate embeddings using a pre-trained model.

    Args:
        expression (pd.DataFrame): Gene expression matrix (cells x genes).
        tokenizer (GraphTokenizer): Tokenizer to tokenize each cell into and compute its edge index.
        cache_dir (str, optional): Directory to cache tokenized data.
    """
    def __init__(self, expression: pd.DataFrame, tokenizer: GraphTokenizer, cache_dir=None):
        self.tokenizer = tokenizer
        self.obs_names = expression.index
        self.expression = expression[expression.columns[expression.columns.isin(self.gene_to_node)]]
        super().__init__(cache_dir=cache_dir, vocab=tokenizer.vocab, mask_fraction=0.0, inference=True)
    
    @property
    def gene_to_node(self):
        return self.tokenizer.gene_to_node

    @property
    def node_to_gene(self):
        return self.tokenizer.node_to_gene
    
    @property
    def network(self):
        return self.tokenizer.network
    
    def __len__(self):
        return len(self.expression)

    def __getitem__(self, idx):
        """
        Returns a tokenized representation of the cell at the given index.

        Args:
            idx (int): Index of the cell.

        Returns:
            dict: Tokenized representation including gene tokens and graph information.
        """
        cell = self.expression.iloc[idx]
        data = self.tokenizer(cell)
        item = self._item_from_tokenized_data(data)
        item["obs_name"] = self.obs_names[idx]
        return item


class VariableNetworksInferenceDataset(InferenceDataset):
    """
    Dataset class for performing inference using cell-specific regulatory networks.

    Each cell is assigned a custom graph (edges and optional weights),
    allowing dynamic modeling of cell-specific regulatory structures.

    Args:
        edge_ids_list (List[np.ndarray]): List of edge indices (one per cell) into `all_edges`.
        all_edges (np.ndarray): Full array of all possible (regulator, target) edges.
        weights_list (List[np.ndarray], optional): List of edge weights per cell.
        limit_regulon (int, optional): Limit number of regulators per target gene.
        limit_graph (int, optional): Limit the total number of edges in the graph.
        drop_unpaired (bool, optional): Whether to drop directed (unpaired) edges when making
            the graph directed.
        **kwargs: Additional arguments passed to `InferenceDataset`.
    """
    def __init__(
            self, 
            edge_ids_list,
            all_edges,
            weights_list=None,
            limit_regulon=None,
            limit_graph=None,
            drop_unpaired=None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.edge_ids_list = edge_ids_list
        self.all_edges = np.array(all_edges)
        self.weights_list = weights_list
        self.limit_regulon = limit_regulon
        self.limit_graph = limit_graph
        self.drop_unpaired = drop_unpaired
    
    @property
    def prune_graph(self):
        return not (self.limit_regulon is None and self.limit_graph is None)

    @property
    def make_undirected(self):
        return self.drop_unpaired is not None

    def __getitem__(self, idx):
        cell = self.expression.iloc[idx]
        edge_ids = self.edge_ids_list[idx]
        edges = self.all_edges[edge_ids]
        regulators, targets = zip(*edges)
        weights = self.weights_list[idx] if self.weights_list else None
        cell_network = RegulatoryNetwork(regulators=regulators, targets=targets, weights=weights, likelihoods=None)

        if self.prune_graph:
            cell_network.prune(limit_regulon=self.limit_regulon, limit_graph=self.limit_graph, inplace=True)
        
        if self.make_undirected:
            cell_network.make_undirected(drop_unpaired=self.drop_unpaired)
        
        data = self.tokenizer(cell, cell_network)
        item = self._item_from_tokenized_data(data)
        item["obs_name"] = self.obs_names[idx]

        return item
    

def get_cell_embeddings(
    dataset: InferenceDataset,
    model: "GDTransformer",
    vocab: GeneVocab = None,
    batch_size=256,
    cls_policy: Literal["include", "exclude", "only"] = "include"
):
    """
    Computes embeddings for each cell in the dataset using a trained GDTransformer model.

    Depending on `cls_policy`, embeddings are derived by:
        - "include": Mean-pooling over all gene nodes, including the CLS token (default).
        - "exclude": Mean-pooling over all gene nodes, excluding the CLS token.
        - "only": Using only the CLS token embedding for each cell.

    Args:
        dataset (InferenceDataset): Dataset containing graph-structured cell inputs.
        model (GDTransformer): Trained transformer model for graph-based gene expression.
        vocab (GeneVocab, optional): Required if `cls_policy` is "exclude" or "only",
            used to identify the CLS token node ID.
        batch_size (int): Number of cells to process per batch.
        cls_policy (str): One of {"include", "exclude", "only"} determining how CLS tokens
            are handled during embedding computation.

    Returns:
        pd.DataFrame: DataFrame of cell-level embeddings with cell names as the index
            and hidden dimensions as columns.
    """
    assert cls_policy in {"include", "exclude", "only"}
    if cls_policy != "include" and vocab is None:
        raise ValueError("vocab must be provided if include_cls is not 'include'")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    x_list = []
    obs_names_list = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Forward Pass"):
            seq_lengths = torch.tensor(batch["num_nodes"], device=device)
            obs_names = batch["obs_name"]
            x = model(send_to_gpu(batch))[0]  # shape: [B, T, H]
            gene_ids = batch["orig_gene_id"]  # shape: [B, T]

            if cls_policy == "only":
                # extract CLS token (assumed to be at position where orig_gene_id == vocab.cls_node)
                cls_mask = (gene_ids == vocab.cls_node).to(x.device)  # [B, T]
                cls_indices = cls_mask.float().argmax(dim=1)  # assume one CLS per cell
                x_cell = x[torch.arange(x.size(0)), cls_indices]  # [B, H]
            else:
                # create mask: optionally exclude CLS
                mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
                if cls_policy == "exclude":
                    cls_mask = (gene_ids == vocab.cls_node).to(x.device)
                    mask = mask & (~cls_mask)
                mask = mask.unsqueeze(-1).float()
                x_masked = x * mask
                x_cell = x_masked.sum(dim=1) / mask.sum(dim=1)

            x_list.append(x_cell.detach().cpu().numpy())
            obs_names_list.append(obs_names)

    return pd.DataFrame(np.vstack(x_list), index=np.concatenate(obs_names_list))


def get_gene_embeddings(
        dataset: InferenceDataset, 
        model: "GDTransformer", 
        vocab: GeneVocab,
        batch_size=256,
        include_cls=False,
    ) -> pd.DataFrame:
    """
    Computes average embeddings for each gene across all cells in the dataset.

    For each gene, this function collects its token embeddings from all cells where
    it is expressed and computes the mean embedding vector.

    Args:
        dataset (InferenceDataset): Dataset containing graph-structured cell inputs.
        model (GDTransformer): Trained model that outputs sequence (node-level) embeddings.
        vocab (GeneVocab, optional): Required to identify and remove the CLS token from the 
            result (if requested) and translate node ids back to gene names.
        batch_size (int): Number of cells to process per batch.
        include_cls (bool): Whether to include the CLS token in the final gene embeddings.
            If False, the CLS embedding (identified via `vocab.cls_node`) is excluded.

    Returns:
        pd.DataFrame: DataFrame with one row per gene (indexed by gene name),
            and one column per hidden dimension from the model.
    """

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=dataset.collate_fn
    )

    gene_embedding_sums = dict()
    gene_counts = defaultdict(int)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Forward Pass"):
            seq_lengths = torch.as_tensor(batch["num_nodes"], device="cuda", dtype=torch.long)
            gene_ids = batch["orig_gene_id"].to("cuda")         # [B, T]
            x = model(send_to_gpu(batch))[0]                    # [B, T, H]

            B, T, H = x.shape
            valid_mask = torch.arange(T, device=x.device)[None, :] < seq_lengths[:, None]  # [B, T]

            flat_x = x.reshape(-1, H)                               # [B*T, H]
            flat_gene_ids = gene_ids.reshape(-1).to(torch.long)     # [B*T]
            flat_mask = valid_mask.reshape(-1)                      # [B*T]

            # Keep only valid tokens
            valid_x = flat_x[flat_mask]                              # [N, H]
            valid_gene_ids = flat_gene_ids[flat_mask]                # [N]

            # Group-by gene_id on GPU
            unique_ids, inverse = torch.unique(valid_gene_ids, return_inverse=True)

            # Sum embeddings per unique gene id
            sums = torch.zeros((unique_ids.numel(), H), device=valid_x.device, dtype=valid_x.dtype)
            sums.index_add_(0, inverse, valid_x)   # segment sum

            # Counts per unique id
            counts = torch.bincount(inverse, minlength=unique_ids.numel())   # [U]

            # Move once to CPU and update Python dicts
            unique_ids_cpu = unique_ids.tolist()
            sums_cpu = sums.cpu().numpy()
            counts_cpu = counts.cpu().tolist()

            for gid, emb_sum, cnt in zip(unique_ids_cpu, sums_cpu, counts_cpu):
                if gid not in gene_embedding_sums:
                    gene_embedding_sums[gid] = emb_sum.copy()
                else:
                    gene_embedding_sums[gid] += emb_sum
                gene_counts[gid] += int(cnt)

    # compute average embedding per gene
    gene_embeddings = {
        gene: gene_embedding_sums[gene] / gene_counts[gene]
        for gene in gene_embedding_sums
    }
    
    if not include_cls:
        gene_embeddings.pop(vocab.cls_node, None)

    df = pd.DataFrame.from_dict(gene_embeddings, orient='index')
    df.index = df.index.map(vocab.node_to_gene)

    return df