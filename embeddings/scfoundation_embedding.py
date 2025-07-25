import argparse
import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import importlib.util

import argparse
import random
import os
from os.path import dirname, abspath, join
import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc

from utils import (
    mask_values, 
    get_locally_indexed_edges, 
    get_locally_indexed_masks_expressions, 
    save_embedding,
    collect_metadata
)

scglm_rootdir = dirname(dirname(abspath(importlib.util.find_spec("scGraphLLM").origin)))
gene_names_map = pd.read_csv(join(scglm_rootdir, "data/gene-name-map.csv"), index_col=0)
ensg2hugo = gene_names_map.set_index("ensg.values")["hugo.values"].to_dict()
hugo2ensg = gene_names_map.set_index("hugo.values")["ensg.values"].to_dict()
ensg2hugo_vectorized = np.vectorize(ensg2hugo.get)
hugo2ensg_vectorized = np.vectorize(hugo2ensg.get)

REG_VALS = "regulator.values"
TAR_VALS = "target.values"


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--retain_obs_vars", nargs="+", default=[])
parser.add_argument("--gene_index_path", type=str, required=True)
parser.add_argument("--scf_rootdir", type=str, required=True)
parser.add_argument("--aracne_dir", type=str, required=True)
parser.add_argument("--sample_n_cells", type=int, default=None)
parser.add_argument("--mask_fraction", type=float, default=None)
parser.add_argument("--mask_value", type=float, default=1e-4)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--cache", action="store_true")
args = parser.parse_args()

# scFoundation imports
sys.path.append(join(args.scf_rootdir, "model"))
from load import *
from get_embedding import main_gene_selection

VOCAB_SIZE = 19264


def gatherData(data, labels, pad_token_id, max_seq_length=None):
    # Step 1: Mask non-expressed genes
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = -float('Inf')

    # Step 2: Add positional bias to prioritize early columns in ties
    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1] - 1, -1, -1)],
                            device=labels.device)
    labels += tmp_data

    # Step 3: Determine how many genes to select
    if max_seq_length is None:
        value_nums = (~none_labels).sum(1)  # number of expressed genes per cell
        max_seq_length = int(value_nums.max().item())

    # Step 4: Pad data and labels to ensure enough entries
    fake_data = torch.full((data.shape[0], max_seq_length), pad_token_id, device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_seq_length), -float('Inf'), device=labels.device)
    labels = torch.hstack([labels, fake_label])

    # Step 5: Select top-k expressed gene indices
    topk_indices = labels.topk(max_seq_length, dim=1).indices

    # Step 6: Gather data and return
    new_data = torch.gather(data, dim=1, index=topk_indices)
    padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels


def get_embedding(pretrainmodel: torch.nn.Module, pretrainconfig, gexpr_feature, input_type, pre_normalized, tgthighres, output_type, pool_type, max_seq_length=None):
    #Inference
    geneexpemb = []
    batchcontainer = []
    position_gene_ids_list = []
    pretrainmodel.eval()
    for i in tqdm(range(gexpr_feature.shape[0])):
        with torch.no_grad():
            #Bulk
            if input_type == 'bulk':
                if pre_normalized == 'T':
                    totalcount = gexpr_feature.iloc[i,:].sum()
                elif pre_normalized == 'F':
                    totalcount = np.log10(gexpr_feature.iloc[i,:].sum())
                else:
                    raise ValueError('pre_normalized must be T or F')
                tmpdata = (gexpr_feature.iloc[i,:]).tolist()
                pretrain_gene_x = torch.tensor(tmpdata+[totalcount,totalcount]).unsqueeze(0).cuda()
                data_gene_ids = torch.arange(VOCAB_SIZE+2, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)
            
            #Single cell
            elif input_type == 'singlecell':  
                expr_series = gexpr_feature.iloc[i, :]
                if pre_normalized == 'A':
                    expr_values = expr_series.iloc[:-1]  # skip totalcount column
                    totalcount = expr_series.iloc[-1]
                else:
                    expr_values = expr_series
                    totalcount = expr_series.sum()

                # # Truncate to top `max_len` expressed genes if specified
                # if max_seq_length is not None and len(expr_values) > max_seq_length:
                #     expr_values = expr_values.nlargest(max_seq_length).sort_index()

                # pre-Normalization
                if pre_normalized == 'F':
                    tmpdata = (np.log1p(gexpr_feature.iloc[i,:]/(gexpr_feature.iloc[i,:].sum())*1e4)).tolist()
                elif pre_normalized == 'T':
                    tmpdata = (gexpr_feature.iloc[i,:]).tolist()
                elif pre_normalized == 'A':
                    tmpdata = (gexpr_feature.iloc[i,:-1]).tolist()
                else:
                    raise ValueError('pre_normalized must be T,F or A')

                if pre_normalized == 'A':
                    totalcount = gexpr_feature.iloc[i,-1]
                else:
                    totalcount = gexpr_feature.iloc[i,:].sum()

                # select resolution
                if tgthighres[0] == 'f':
                    pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount*float(tgthighres[1:])),np.log10(totalcount)]).unsqueeze(0).cuda()
                elif tgthighres[0] == 'a':
                    pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount)+float(tgthighres[1:]),np.log10(totalcount)]).unsqueeze(0).cuda()
                elif tgthighres[0] == 't':
                    pretrain_gene_x = torch.tensor(tmpdata+[float(  tgthighres[1:]),np.log10(totalcount)]).unsqueeze(0).cuda()
                else:
                    raise ValueError('tgthighres must be start with f, a or t')
                
                data_gene_ids = torch.arange(VOCAB_SIZE+2, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

            # data_gene_ids = torch.arange(VOCAB_SIZE+2, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)
            value_labels = pretrain_gene_x > 0
            x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'], max_seq_length=max_seq_length)
            # position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
            
            # raw embedding tensor
            if output_type=='raw':
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'], max_seq_length=max_seq_length)
                position_gene_ids_list.append(position_gene_ids.detach().cpu().numpy())
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
                position_emb = pretrainmodel.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = pretrainmodel.encoder(x,x_padding)
                geneexpemb.append(geneemb.detach().cpu().numpy())

            #Cell embedding
            elif output_type=='cell':
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
                position_emb = pretrainmodel.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = pretrainmodel.encoder(x,x_padding)

                geneemb1 = geneemb[:,-1,:]
                geneemb2 = geneemb[:,-2,:]
                geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
                geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
                if pool_type=='all':
                    geneembmerge = torch.concat([geneemb1,geneemb2,geneemb3,geneemb4],axis=1)
                elif pool_type=='max':
                    geneembmerge, _ = torch.max(geneemb, dim=1)
                else:
                    raise ValueError('pool_type must be all or max')
                geneexpemb.append(geneembmerge.detach().cpu().numpy())

            #Gene embedding
            elif output_type=='gene':
                pretrainmodel.to_final = None
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(),pretrain_gene_x.float(),pretrainconfig)
                out = pretrainmodel.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                            )
                out = out[:,:VOCAB_SIZE,:].contiguous()
                geneexpemb.append(out.detach().cpu().numpy())

            #Gene batch embedding
            elif output_type=='gene_batch':
                batchcontainer.append(pretrain_gene_x.float())
                if len(batchcontainer)==gexpr_feature.shape[0]:
                    batchcontainer = torch.concat(batchcontainer,axis=0)
                else:
                    continue
                pretrainmodel.to_final = None
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(batchcontainer,batchcontainer,pretrainconfig)
                out = pretrainmodel.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                            )
                geneexpemb = out[:,:VOCAB_SIZE,:].contiguous().detach().cpu().numpy()
            #Gene_expression
            elif output_type=='gene_expression':
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(),pretrain_gene_x.float(),pretrainconfig)
                out = pretrainmodel.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                            )
                out = out[:,:VOCAB_SIZE].contiguous()
                geneexpemb.append(out.detach().cpu().numpy())                
            else:
                raise ValueError('output_type must be cell or gene or gene_batch or gene_expression or raw')
    
    if output_type == "raw":
        # Remove last two tokens
        trimmed_geneexpemb = [emb[:, :-2, :] for emb in geneexpemb]
        max_seq_length = max(emb.shape[1] for emb in trimmed_geneexpemb)
        embeddings = np.concatenate([
            np.pad(emb, pad_width=((0, 0), (0, max_seq_length - emb.shape[1]), (0, 0)), 
                   mode="constant", constant_values=0)
            for emb in trimmed_geneexpemb
        ], axis=0)
        gene_ids_list = [gene_ids[:,:-2] for gene_ids in position_gene_ids_list]
        return embeddings, gene_ids_list

    geneexpemb = np.squeeze(np.array(geneexpemb))
    return geneexpemb


def load_data(data_path, vocab_size, gene_list, pre_normalized, input_type, demo=False):
    """Loads gene expression data from various formats and preprocesses it."""
    if data_path.endswith('npz'):
        gexpr_feature = scipy.sparse.load_npz(data_path).toarray()
        gexpr_feature = pd.DataFrame(gexpr_feature)
    elif data_path.endswith('h5ad'):
        gexpr_feature = sc.read_h5ad(data_path)
        idx = gexpr_feature.obs_names.tolist()
        col = getattr(gexpr_feature.var, 'gene_name', gexpr_feature.var_names).tolist()
        gexpr_feature = gexpr_feature.X.toarray() if issparse(gexpr_feature.X) else gexpr_feature.X
        gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
    elif data_path.endswith('npy'):
        gexpr_feature = pd.DataFrame(np.load(data_path))
    else:
        gexpr_feature = pd.read_csv(data_path, index_col=0)
    
    gexpr_feature = preprocess_data(gexpr_feature, vocab_size, gene_list, pre_normalized, input_type, demo)
    
    return gexpr_feature

def preprocess_data(gexpr_feature, vocab_size, gene_list, pre_normalized, input_type, demo=False):
    if gexpr_feature.shape[1] < vocab_size:
        print('Converting gene feature into VOCAB_SIZE')
        gexpr_feature, _, _ = main_gene_selection(gexpr_feature, gene_list)
        assert gexpr_feature.shape[1] >= vocab_size
    
    if (pre_normalized == 'F') and (input_type == 'bulk'):
        adata = sc.AnnData(gexpr_feature)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        gexpr_feature = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    
    if demo:
        gexpr_feature = gexpr_feature.iloc[:10, :]
    return gexpr_feature


def load_model(model_path, version, output_type, rootdir):
    """Loads the pre-trained model from the given checkpoint."""
    if version == 'noversion':
        ckpt_path = model_path
        key = None
    else:
        ckpt_path = join(rootdir, 'model/models/models.ckpt')
        if output_type in ['cell', 'raw']:
            key = 'cell' if version == 'ce' else 'rde'
        elif output_type in ['gene', 'gene_batch', 'gene_expression']:
            key = 'gene'
        else:
            raise ValueError('output_mode must be one of cell, gene, gene_batch, gene_expression')
    
    return load_model_frommmf(ckpt_path, key)


def main(args):
    # Load cells & translate to human symbol gene names
    data = sc.read_h5ad(args.cells_path)

    if args.sample_n_cells is not None and data.n_obs > args.sample_n_cells:
        sc.pp.subsample(data, n_obs=args.sample_n_cells, random_state=12345, copy=False)
    
    data_original = data.copy()
    ensg2hugo_map = data_original.var_names.to_series().apply(ensg2hugo.get)
    hugo2ensg_map = {v: k for k, v in ensg2hugo_map.items() if pd.notna(v)}

    if args.mask_fraction is not None:
        X_masked, masked_indices = mask_values(data.X.astype(float), mask_prob=args.mask_fraction, mask_value=args.mask_value)
        data.X = X_masked

    # convert to raw counts
    # counts = sc.AnnData(
    #     X=csc_matrix(adata.layers["counts"].astype(int)),
    #     obs=adata.obs[["n_counts"]],
    #     var=pd.DataFrame(index=adata.var.index).assign(**{"ensembl_id": lambda df: df.index.to_series()}),
    # )

    # data.var["symbol_id"] = data.var_names.to_series().apply(ensg2hugo.get)
    data.var["symbol_id"] = ensg2hugo_map
    # data = data[:, ~data.var["symbol_id"].isna()]
    data.var.set_index("symbol_id")
    data.var_names = data.var["symbol_id"]

    data_df = data.to_df()

    # load scFoundation gene index
    gene_list_df = pd.read_csv(args.gene_index_path, header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])

    # Load data
    gexpr_feature = preprocess_data(
        gexpr_feature=data_df, 
        vocab_size=VOCAB_SIZE,
        gene_list=gene_list, 
        pre_normalized=True, 
        input_type="singlecell", 
        demo=False
    )

    # Load model
    pretrainmodel, pretrainconfig = load_model(
        model_path=args.model_path, 
        version="rde", 
        output_type="raw", 
        rootdir=args.scf_rootdir
    )
    
    # gene_ids is a list of lists, where element i,j, is the 
    # expression value of gene j (according to gexpr_feature) in cell i
    embeddings, symbol_ids = get_embedding(
        pretrainmodel=pretrainmodel,
        pretrainconfig=pretrainconfig,
        gexpr_feature=gexpr_feature,
        input_type="singlecell",
        pre_normalized="T",
        tgthighres="t4",
        output_type="raw",
        pool_type=None,
        max_seq_length=args.max_seq_length + 2 # add 2 to account for special tokens
    )
    
    id_symbol_map = pd.Series(gexpr_feature.columns).to_dict()
    id_gene_map_vectorized = np.vectorize(lambda x: id_symbol_map.get(x))
    genes_list = [id_gene_map_vectorized(ids) for ids in symbol_ids]
    seq_lengths = [g.shape[1] for g in genes_list]
    max_seq_length = max(seq_lengths)
    assert max_seq_length == embeddings.shape[1], "max sequence length differs between genes ids and embedding tensor"
    genes_symbol = np.concatenate([
        np.pad(g, pad_width=((0, 0), (0, max_seq_length - g.shape[1])), 
               mode="constant", constant_values="<pad>")
        for g in genes_list
    ], axis=0)
    # genes_ensg = [hugo2ensg_vectorized(genes) for genes in genes_list]
    # genes_ensg = hugo2ensg_vectorized(genes_symbol)
    genes_ensg = np.array([
        [hugo2ensg_map.get(gene, None) for gene in row]
        for row in genes_symbol
    ])

    original_genes = set(data_original.var_names)
    backward_genes = set(list(genes_ensg.flatten()))
    # Compute intersection and union
    intersection = original_genes & backward_genes
    union = original_genes | backward_genes
    print(f"Jaccard: {len(intersection) / len(union)}")
    print(f"Original Only: {len(original_genes - backward_genes)}")
    print(f"Backward Only: {len(backward_genes - original_genes)}")
    print(f"Intersection: {len(intersection)}")


    # load aracne network
    network = pd.read_csv(join(args.aracne_dir, "consolidated-net_defaultid.tsv"), sep="\t")
    edges = get_locally_indexed_edges(genes_ensg, src_nodes=network[REG_VALS], dst_nodes=network[TAR_VALS])
    
    # gene name map is not bijective between ENSG and HUGO, and some genes are getting
    # incorrectly backward translated
    # get original expression
    # expression = np.concatenate([
    #     np.pad(data_original[i, genes[:seq_lengths[i]]].X.toarray(), 
    #            pad_width=((0,0), (0, max_seq_length - seq_lengths[i])), 
    #            mode="constant", constant_values=0)
    #     for i, genes in enumerate(genes_ensg)
    # ], axis=0)

    # get metadata
    metadata = collect_metadata(data, args.retain_obs_vars)

    if args.mask_fraction is None:
        save_embedding(
            file=args.emb_path,
            cache=args.cache,
            cache_dir=args.emb_cache,
            x=embeddings,
            seq_lengths=seq_lengths,
            # expression=expression,
            edges=edges,
            metadata=metadata
        )
        return
    
    masks, masked_expressions = get_locally_indexed_masks_expressions(data_original, masked_indices, genes_ensg)
    save_embedding(
        file=args.emb_path,
        cache=args.cache,
        cache_dir=args.emb_cache,
        x=embeddings,
        seq_lengths=seq_lengths,
        # expression=expression,
        edges=edges,
        metadata=metadata,
        masks=masks,
        masked_expressions=masked_expressions
    )                
    

if __name__ == "__main__":
    args.cells_path = join(args.data_dir, "cells.h5ad")
    args.emb_path = join(args.out_dir, "embedding.npz")
    args.emb_cache = join(args.out_dir, "cached_embeddings")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.cache:
        os.makedirs(args.emb_cache, exist_ok=True)
    main(args)
