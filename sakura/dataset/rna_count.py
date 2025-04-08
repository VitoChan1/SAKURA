"""
General scRNA-Seq count dataset
"""

import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sakura.utils.data_transformations import ToKBins
# Transformations
from sakura.utils.data_transformations import ToOnehot
from sakura.utils.data_transformations import ToOrdinal
from sakura.utils.data_transformations import ToTensor


class SCRNASeqCountData(Dataset):
    """
    General scRNA-Seq count dataset class for SAKURA inputs

    :param gene_csv_path:  Path to the gene csv file
    :type gene_csv_path: str
    :param pheno_csv_path: Path to the phenotype csv file
    :type pheno_csv_path: str
    :param gene_meta_json_path: Path to the genotype meta JSON file
    :type gene_meta_json_path: str, optional
    :param pheno_meta_json_path: Path to the phenotype meta JSON file
    :type pheno_meta_json_path: str, optional
    :param pheno_df_dtype: Pandas dtype applied to phenotype data,
        either the whole dataframe or individual columns
    :type pheno_df_dtype: dtype or dict of {Hashable dtype}, optional
    :param pheno_df_na_filter*: Detect missing value markers
        (empty strings and the value of na_values), defaults to True
        :type pheno_df_na_filter: bool
    :param gene_meta*: A configuration dictionary related to gene data processing
    :type gene_meta: dict[str, Any], optional
    :param pheno_meta: A dictionary contains definition and configurations of phenotype data
    :type pheno_meta: dict[str, Any], optional
    :param mode: data export option ['all','key', or others] of the dataset, defaults to 'all'.
    :type mode: str
    :param verbose: Whether to enable verbose console logging, defaults to False
    :type verbose: bool

    **Expected inputs:**

    gene_csv:
        • Assuming rows are genes, columns are samples (or cells)
        • rownames are gene identifiers (gene names, or Ensembl IDs)
        • colnames are sample identifiers (cell names)
    genotype_meta_csv:
        • A JSON file related to gene data processing
        • pre_procedure: transformations that will perform when *load* the dataset
        • post_procedure: transformations that will perform when *export* requested samples
    phenotype_csv:
        • Assuming rows are samples, columns are metadata contents
        • rownames are sample identifiers (cell names)
    phenotype_meta_csv:
        • A JSON file to define Type, Range, and Order for phenotype columns, and related to phenotype configurations for SAKURA model training
        • Storage entity is a dict
        • Type: 'categorical', 'numeric', 'ordinal' (tbd)
        • For 'categorical' range: array of possible values, *ordered*
        • pre_procedure: transformations that will perform when *load* the dataset
        • post_procedure: transformations that will perform when *export* requested samples
    Modes:
        • 'all': export both raw and processed data, together with names/keys of cells
        • 'key': export only names/keys of cells
        • otherwise: export only processed data
    Transformations:
        • ToTensor: convert input data into a PyTorch tensor; input type should be 'gene' or 'pheno'
        • ToOneHot: transform categorical data to one-hot encoding; an order of classes should be specified, otherwise will use sorted labels, assuming the range of labels is derived from input data
        • ToOrdinal: convert categorical data into ordinal (integer) encoding; each unique category is assigned with a unique integer value, which can be useful for models that require numerical input
        • ToKBins: transform continuous data into `k` bins; quantile-based binning is applied to convert continuous features into categorical features

    .. note::
        For more details of the transformations, see :func:`utils.data_transformations`.

        **<gene_meta> example:**

        .. code-block::

            {
                'all': {
                    'gene_list': "*",
                    'pre_procedure': [],
                    'post_procedure': [{
                    'type': "ToTensor"
                    }]
                }
            }

        **<pheno_meta>:** For more details of the JSON structure, see :func:`utils.data_transformations`.

        **<na_filter>:** For phenotype data without any NA values, passing <na_filter>=False can
        improve the performance of reading a large file.
    """

    def __init__(self, gene_csv_path, pheno_csv_path,
                 gene_meta_json_path=None, pheno_meta_json_path=None,
                 pheno_df_dtype=None, pheno_df_na_filter=True,
                 gene_meta=None, pheno_meta=None,
                 mode='all', verbose=False):
        # Verbose console logging
        self.verbose=verbose

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
        self._gene_expr_mat_orig = pd.read_csv(self.gene_csv_path, index_col=0, header=0)
        self.gene_expr_mat = self._gene_expr_mat_orig.copy()

        if self.verbose:
            print('==========================')
            print('rna_count dataset:')
            print("Imported gene expression matrix CSV from:", self.gene_csv_path)
            print(self.gene_expr_mat.shape)
            print(self.gene_expr_mat.head(3))

        # Read gene expression matrix metadata
        self.gene_meta = gene_meta
        self.flag_expr_set_pre_sliced = False
        self.expr_mat_pre_sliced = dict()
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
        self._pheno_df_orig = pd.read_csv(self.pheno_csv_path, index_col=0, header=0, dtype=pheno_df_dtype, na_filter=pheno_df_na_filter)
        self.pheno_df = self._pheno_df_orig.copy()

        if self.verbose:
            print("Phenotype data from CSV from:", self.pheno_csv_path)
            print(self.pheno_df.shape)
            print(self.pheno_df)

        # Read phenotype colmun metadata
        self.pheno_meta = pheno_meta
        if self.pheno_meta is None:
            self.pheno_meta = {}
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

    def expr_set_pre_slice(self):
        """
        Pre-slices the expression matrix based on gene metadata.

        This function iterates over the keys in the <gene_meta> dictionary, and for each key,
        it retrieves the corresponding metadata to determine how to slice the expression matrix.
        The results are stored in the <expr_mat_pre_sliced> attribute.

        gene_list:
            • list or numpy array: the function extracts the corresponding rows from the `gene_expr_mat`;
            • '*': the entire expression matrix is copied;
            • '-': the function drops the rows specified in <exclude_list> from the expression matrix.

        After execution, the flag <flag_expr_set_pre_sliced> is set to True.

        :return: None
        """

        self.flag_expr_set_pre_sliced = True
        self.expr_mat_pre_sliced = dict()
        for cur_expr_key in self.gene_meta.keys():
            cur_expr_meta = self.gene_meta[cur_expr_key]
            if type(cur_expr_meta['gene_list']) is list or type(cur_expr_meta['gene_list']) is np.ndarray:
                self.expr_mat_pre_sliced[cur_expr_key] = self.gene_expr_mat.loc[cur_expr_meta['gene_list'], :].copy()
            elif cur_expr_meta['gene_list'] == '*':
                self.expr_mat_pre_sliced[cur_expr_key] = self.gene_expr_mat.copy()
            elif cur_expr_meta['gene_list'] == '-':
                self.expr_mat_pre_sliced[cur_expr_key] = self.gene_expr_mat.drop(cur_expr_meta['exclude_list'],
                                                                                 axis=0).copy()

    def __select_expr_mat(self, cur_expr_key, item):
        cur_expr_mat = None
        if self.flag_expr_set_pre_sliced:
            # Select pre-sliced expression matrices
            cur_expr_mat = self.expr_mat_pre_sliced[cur_expr_key].iloc[:, item].copy()
        else:
            # Ad-lib slice expression matrix
            cur_expr_meta = self.gene_meta[cur_expr_key]
            # Gene Selection
            if type(cur_expr_meta['gene_list']) is list or type(cur_expr_meta['gene_list']) is np.ndarray:
                # Select given genes
                cur_expr_mat = self.gene_expr_mat.loc[cur_expr_meta['gene_list'], :].iloc[:, item].copy()
            elif cur_expr_meta['gene_list'] == '*':
                # Select all genes
                cur_expr_mat = self.gene_expr_mat.iloc[:, item].copy()
            elif cur_expr_meta['gene_list'] == '-':
                # Deselect given genes
                cur_expr_mat = self.gene_expr_mat.drop(cur_expr_meta['exclude_list'], axis=0).iloc[:, item].copy()
        return cur_expr_mat

    def export_data(self, item,
                    include_raw=True,
                    include_proc=True,
                    include_cell_key=True):
        """
        Export a batch of data based on the specified items and expression data flags.

        :param item: The index to select data from the dataset
        :type item: list or int
        :param include_raw: Whether to export the unprocessed, subset expression matrix and
            phenotype data frame, defaults to True
        :type include_raw: bool
        :param include_proc: Whether to export the processed data (following procedures specified in the configs),
            defaults to True
        :type include_proc: bool
        :param include_cell_key: Should the names/keys of the cells be exported
        :type include_cell_key: bool

        :return: A dictionary containing data of the specified items from the dataset
        :rtype: dict[str, Any]
        """

        # Type adaptation: when 'item' is a single index, convert it to list
        if type(item) is int:
            item = [item]

        # Extract required cells and prepare required structure
        ret = dict()

        if include_cell_key is True:
            ret['cell_key'] = self.gene_expr_mat.columns.values[item]

        # Prepare raw gene output (if needed)
        if include_raw is True:
            ret['expr_mat'] = self.gene_expr_mat.iloc[:, item].copy()
        if include_proc is True:
            ret['expr'] = dict()
            for cur_expr_key in self.gene_meta.keys():
                cur_expr_meta = self.gene_meta[cur_expr_key]
                ret['expr'][cur_expr_key] = self.__select_expr_mat(cur_expr_key, item)
                # Post Transformation
                for cur_procedure in cur_expr_meta['post_procedure']:
                    if cur_procedure['type'] == 'ToTensor':
                        ret['expr'][cur_expr_key] = self.to_tensor(ret['expr'][cur_expr_key],
                                                                   input_type='gene',
                                                                   force_tensor_type=cur_procedure.get('force_tensor_type'))
                    else:
                        print("Unsupported post-transformation")
                        raise NotImplementedError

        # Prepare phenotype output
        if include_raw is True:
            ret['pheno_df'] = self.pheno_df.iloc[item, :].copy()
        if include_proc is True:
            ret['pheno'] = dict()
            for pheno_output_key in self.pheno_meta.keys():
                cur_pheno_meta = self.pheno_meta[pheno_output_key]

                # Slice phenotype dataframe
                if cur_pheno_meta['type'] == 'categorical':
                    ret['pheno'][pheno_output_key] = self.pheno_df.loc[:, [cur_pheno_meta['pheno_df_key']]].iloc[item, :].copy()
                    # Process phenotype label as required
                    for cur_procedure in cur_pheno_meta['post_procedure']:
                        if cur_procedure['type'] == 'ToTensor':
                            ret['pheno'][pheno_output_key] = self.to_tensor(sample=ret['pheno'][pheno_output_key],
                                                                            input_type='pheno',
                                                                            force_tensor_type=cur_procedure.get('force_tensor_type'))
                        elif cur_procedure['type'] == 'ToOnehot':
                            ret['pheno'][pheno_output_key] = self.to_onehot(sample=ret['pheno'][pheno_output_key],
                                                                            order=cur_pheno_meta['order'])
                        elif cur_procedure['type'] == 'ToOrdinal':
                            ret['pheno'][pheno_output_key] = self.to_ordinal(sample=ret['pheno'][pheno_output_key],
                                                                             order=cur_pheno_meta['order'])
                        elif cur_procedure['type'] == 'ToKBins':
                            ret['pheno'][pheno_output_key] = self.to_kbins(sample=ret['pheno'][pheno_output_key],
                                                                           n_bins=cur_procedure['n_bins'],
                                                                           encode=cur_procedure['encode'],
                                                                           strategy=cur_procedure['strategy'])
                        else:
                            raise NotImplementedError('Unsupported transformation type for phenotype groups')
                elif cur_pheno_meta['type'] == 'numerical':
                    ret['pheno'][pheno_output_key] = self.pheno_df.loc[:, cur_pheno_meta['pheno_df_keys']].iloc[item, :].copy()
                    # Process phenotype label as required
                    for cur_procedure in cur_pheno_meta['post_procedure']:
                        if cur_procedure['type'] == 'ToTensor':
                            ret['pheno'][pheno_output_key] = self.to_tensor(sample=ret['pheno'][pheno_output_key],
                                                                            input_type='pheno',
                                                                            force_tensor_type=cur_procedure.get('force_tensor_type'))
                        else:
                            raise NotImplementedError('Unsupported transformation type for phenotype groups')
                else:
                    raise ValueError('Unsupported phenotype group name')

        return ret

    def __len__(self):
        # Length of the dataset is considered as the number of cells
        return self.gene_expr_mat.shape[1]

    def __getitem__(self, item):
        if self.mode == 'all':
            return self.export_data(item,
                                    include_raw=True,
                                    include_proc=True,
                                    include_cell_key=True)
        elif self.mode == 'key':
            return self.export_data(item,
                                    include_raw=False,
                                    include_proc=False,
                                    include_cell_key=True)
        else:
            return self.export_data(item,
                                    include_raw=False,
                                    include_proc=True,
                                    include_cell_key=False)
