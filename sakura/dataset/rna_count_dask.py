"""
Dask version of scRNA-seq count data
"""

import json

import dask.dataframe as dd
import pandas as pd
from torch.utils.data import Dataset

from sakura.utils.data_transformations import ToKBins
# Transformations
from sakura.utils.data_transformations import ToOnehot
from sakura.utils.data_transformations import ToOrdinal
from sakura.utils.data_transformations import ToTensor


class SCRNASeqCountDataDask(Dataset):
    """
    Dask version of scRNA-seq count datas. Fits for dataset with a very large number of cells.

    Parameters
    ----------
    gene_csv:
        * Assuming rows are genes, colmuns are samples(/cells)
        * rownames are gene identifiers (gene name, or ensembl ID)
        * colnames are sample identifiers (cell name)
    genotype_meta_csv:
        * pre_procedure: transformations that will perform when *load* the dataset
        * post_procedure: transformations that will perform when *export* requested samples
    phenotype_csv:
        * Assuming rows are samples, columns are metadata contents
        * rownames are sample identifiers ()
    phenotype_meta_csv:
        * A json file to define Type, Range, and Order for phenotype columns
        * Storage entity is a 'dict'
        * Type: 'categorical', 'numeric', 'ordinal' (tbd)
        * For 'categorical':
            * Range: array of possible values, *ordered*
        * pre_procedure
        * post_procedure


    Options:
        * Mode

    Modes:
        * all
        * sample_id
        * expr
        * pheno



    Transformations:
        * ToTensor:
        * ToOneHot: transform categorical data to one-hot encoding, an order of classes should be specified, otherwise
                    will use sorted labels, assuming the range of labels are from input
        * ToOrdinal:
        * ToKBins:
        * LogNormalize:
    Gene expression matrix: rows are cells, columns are genes(Unlike rna_count, which directly accepts the Seurat compatible datasheets (i.e. row gene, col cell))
    Phenotype matrix: rows are cells, columns are features

    """

    def __init__(self, gene_csv_path, pheno_csv_path,
                 gene_meta_json_path=None, pheno_meta_json_path=None,
                 gene_meta=None, pheno_meta=None,
                 mode='all', verbose=False):

        # Verbose console logging
        self.verbose = verbose

        # Persist argument list
        self.gene_csv_path = gene_csv_path
        self.gene_meta_json_path = gene_meta_json_path
        self.pheno_csv_path = pheno_csv_path
        self.pheno_meta_json_path = pheno_meta_json_path
        self.mode = mode

        # Register transformers
        self.to_tensor = ToTensor()
        self.to_onehot = ToOnehot()
        self.to_ordinal = ToOrdinal()
        self.to_kbins = ToKBins()

        # Read gene expression matrix

        self._gene_expr_mat_orig = dd.read_csv(self.gene_csv_path)
        self.gene_expr_mat = self._gene_expr_mat_orig.copy()

        if self.verbose:
            print('==========================')
            print('Dask version of rna_count dataset:')
            print("Imported gene expression matrix CSV from:", self.gene_csv_path)
            print(self.gene_expr_mat.shape)
            print(self.gene_expr_mat.head(3))

        # Read gene expression matrix metadata
        self.gene_meta = gene_meta
        if self.gene_meta is None:
            self.gene_meta = {
                "all": {
                    "gene_list": "*",
                    "pre_procedure": [],
                    'post_procedure': [
                        {
                            "type": "ToTensor"
                        }
                    ]
                }
            }
            if self.verbose:
                print('No external gene expression set provided, using dummy.')
        if self.gene_meta_json_path is not None:
            with open(self.gene_meta_json_path, 'r') as f:
                self.gene_meta = json.load(f)
                if self.verbose:
                    print("Gene expression set metadata imported from:", self.gene_meta_json_path)
        if self.verbose:
            print("Gene expression set metadata:")
            print(self.gene_meta)

        # Read phenotype data frame
        self._pheno_df_orig = pd.read_csv(self.pheno_csv_path, index_col=0, header=0)
        self.pheno_df = self._pheno_df_orig.copy()

        if self.verbose:
            print("Phenotype data from CSV from:", self.pheno_csv_path)
            print(self.pheno_df.shape)
            print(self.pheno_df)

        # Read phenotype colmun metadata
        self.pheno_meta = pheno_meta
        if pheno_meta_json_path is not None:
            if self.verbose:
                print("Reading phenotype metadata json from:", self.pheno_meta_json_path)
            with open(self.pheno_meta_json_path, 'r') as f:
                self.pheno_meta = json.load(f)

        # Cell list
        self._cell_list_orig = self._gene_expr_mat_orig.columns.values
        self.cell_list = self._cell_list_orig.copy()

        # Sanity check
        # Cell should be consistent between expr matrix and phenotype table
        if (self._gene_expr_mat_orig.columns.values != self._pheno_df_orig.index.values).any():
            raise ValueError
