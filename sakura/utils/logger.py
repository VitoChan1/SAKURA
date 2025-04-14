"""
Logger of SAKURA pipeline
"""

import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import tensorboardX


class Logger(object):
    """
    SAKURA Pipeline Logging Controller

    Handles multi-dimensional logging including:

        - Real-time loss tracking via TensorBoard
        - Latent space embedding exports
        - Model configuration persistence

    :param log_path: Root directory for all logging artifacts, defaults to './logs/'
    :type log_path: str, optional
    :param suppress_tensorboardX: Whether to disable TensorBoard integration, defaults to False
    :type suppress_tensorboardX: bool, optional
    """
    def __init__(self, log_path='./logs/', suppress_tensorboardX=False):

        # Arguments
        self.log_path = log_path

        # Create logging directory if not exist
        Path(log_path).mkdir(parents=True, exist_ok=True)

        # SummaryWriter from tensorboardX
        self.log_writer = None
        if not suppress_tensorboardX:
            self.log_writer = tensorboardX.SummaryWriter(self.log_path)



    def log_loss(self, trainer_output: dict, tick,
                 loss_name_prefix='',
                 selected_loss_group=['loss', 'regularization']):
        """
        Record hierarchical loss structure to TensorBoard

        :param trainer_output:
            Nested loss structure containing:
            - pheno_loss: Dictionary of phenotype-specific losses
            - signature_loss: Dictionary of signature-specific losses
            - main_latent_loss: Core latent space loss components
        :type trainer_output:
        :param tick: Training iteration counter for x-axis scaling
        :type tick: int
        :param loss_name_prefix: Namespace prefix for loss grouping, defaults to ''
        :type loss_name_prefix: str, optional
        :param selected_loss_group: Loss types to log from ['loss', 'regularization'], defaults to both
        :type selected_loss_group: list, optional

        :return: None
        """

        # TODO: Selectively log losses

        # Class label supervision loss and regularization loss
        for cur_pheno in trainer_output['pheno_loss'].keys():
            for cur_group in selected_loss_group:
                for cur_loss_key in trainer_output['pheno_loss'][cur_pheno][cur_group].keys():
                    self.log_writer.add_scalar(loss_name_prefix + '_pheno_' + cur_pheno + '_' + cur_group + '_' + cur_loss_key,
                                               trainer_output['pheno_loss'][cur_pheno][cur_group][cur_loss_key],
                                               tick)


        # Signature supervision loss
        for cur_signature in trainer_output['signature_loss'].keys():
            for cur_group in selected_loss_group:
                for cur_loss_key in trainer_output['signature_loss'][cur_signature][cur_group].keys():
                    self.log_writer.add_scalar(
                        loss_name_prefix + '_signature_' + cur_signature + '_' + cur_group + '_' + cur_loss_key,
                        trainer_output['signature_loss'][cur_signature][cur_group][cur_loss_key],
                        tick)

        # Main latent loss
        for cur_group in selected_loss_group:
            for cur_loss_key in trainer_output['main_latent_loss'][cur_group].keys():
                self.log_writer.add_scalar(loss_name_prefix + '_main_latent_' + cur_group + '_' + cur_loss_key,
                                           trainer_output['main_latent_loss'][cur_group][cur_loss_key],
                                           tick)

    def log_parameter(self, trainer_output: dict, tick,
                      log_prefix=''):
        """
        ( to be implemented )
        """
        # TODO: log parameters on Tensorboard
        raise NotImplementedError

    def log_metric(self, trainer_output: dict, tick, metric_configs: dict,
                   log_prefix=''):
        """
        ( to be implemented )
        """
        # TODO: log metrics (AUROC, AUPR, etc.) on Tensorboard
        for cur_metric in metric_configs:
            if cur_metric['type'] == 'AUROC':
                raise NotImplementedError
            elif cur_metric['type'] == 'AUPR':
                # Log AUPR only
                raise NotImplementedError
            elif cur_metric['type'] == 'PR':
                # Log P, R, AUPR, and PR Curve
                raise NotImplementedError
            elif cur_metric['type'] == 'ROC':
                # Log ROC Curve
                raise NotImplementedError
            else:
                raise NotImplementedError

    def dump_latent_to_csv(self, controller_output,
                           dump_main=True,
                           dump_lat_pre=False,
                           dump_pheno=True, dump_pheno_out=False, selected_pheno=None,
                           dump_signature=True, dump_signature_out=False, selected_signature=None,
                           dump_re_x=False, re_x_col_naming='dimid',
                           rownames=None, colnames=None,
                           path='latent.csv',
                           compression='none'):
        """
        Export latent representations from the output of the controller to structured storage.

        :param controller_output: Forward pass results containing latent components returned by the controller
        :type controller_output: dict
        :param dump_main: Whether to export main latent space, defaults to True
        :type dump_main: bool, optional
        :param dump_lat_pre: Whether to export pre-encoder latent space, defaults to False
        :type dump_re_x: bool, optional
        :param dump_pheno: Whether to export phenotype latent space, defaults to True
        :type dump_pheno: bool, optional
        :param dump_pheno_out: Whether to export output of the phenotype module, defaults to False
        :type dump_pheno_out: bool, optional
        :param selected_pheno: A list containing phenotype ids to be dumped
        :type selected_pheno: list[str]
        :param dump_signature: Whether to export signature latent space, defaults to False
        :type dump_signature: bool, optional
        :param dump_signature_out: Whether to export output of the signature module, defaults to False
        :type dump_signature_out: bool, optional
        :param selected_signature: a list containing signature ids to be dumped
        :type selected_signature: list[str]
        :param dump_re_x: Whether to export output of the reconstruction module, defaults to False
        :type dump_re_x: bool, optional
        :param re_x_col_naming: Set the column names of the reconstructed matrix,
            can be 'dimid' (by default) which means dimension id or 'genenames' (names of genes)
        :type re_x_col_naming: Literal['dimid', 'genenames'], optional
        :param colnames: Column label index to assign to DataFrame
        :type colnames: list-like
        :param rownames: Row label index to assign to DataFrame
        :type rownames: list-like
        :param path: Output path for the CSV file storing, defaults to 'latent.csv'
        :type path: str, optional
        :param compression: compression type of CSV files, can be 'hdf', 'gzip' or 'none' (by default)
        :type compression: Literal['none', 'gzip', 'hdf'], optional

        :return: None
        """

        lat_all = list()
        # Main latent (lat_main)
        if dump_main:
            lat_all.append(
                pd.DataFrame(controller_output['fwd_res']['lat_main'].numpy()).set_axis(['main.' + str(i + 1) for i in range(
                    controller_output['fwd_res']['lat_main'].shape[1]
                )], axis=1, inplace=False)
            )

        # Reconstructed input
        # Matrix to dump may be very large
        if dump_re_x:
            if re_x_col_naming == 'dimid':
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['re_x'].numpy()).set_axis(['re_x.' + str(i + 1) for i in range(
                        controller_output['fwd_res']['re_x'].shape[1]
                    )], axis=1, inplace=False)
                )
            elif re_x_col_naming == 'genenames':
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['re_x'].numpy()).set_axis(colnames, axis=1, inplace=False)
                )

        # Pre-encoder results (lat_pre)
        if dump_lat_pre:
            lat_all.append(
                pd.DataFrame(controller_output['fwd_res']['lat_pre'].numpy()).set_axis(['lat_pre.' + str(i + 1) for i in range(
                    controller_output['fwd_res']['lat_pre'].shape[1]
                )], axis=1, inplace=False)
            )

        # Phenotype latents
        if dump_pheno:
            if selected_pheno is None:
                selected_pheno = controller_output['fwd_res']['lat_pheno'].keys()
                warnings.warn(message='Phenotype selection ot specified, using: ' + str(selected_pheno))
            for cur_pheno in selected_pheno:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['lat_pheno'][cur_pheno].numpy()).set_axis(
                        [cur_pheno + '.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['lat_pheno'][cur_pheno].shape[1]
                        )], axis=1, inplace=False)
                )

        # Phenotype outputs
        if dump_pheno_out:
            if selected_pheno is None:
                selected_pheno = controller_output['fwd_res']['pheno_out'].keys()
                warnings.warn(message='Phenotype selection ot specified, using: ' + str(selected_pheno))
            for cur_pheno in selected_pheno:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['pheno_out'][cur_pheno].numpy()).set_axis(
                        [cur_pheno + '.out.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['pheno_out'][cur_pheno].shape[1]
                        )], axis=1, inplace=False)
                )

        # Signature latents
        if dump_signature:
            if selected_signature is None:
                selected_pheno = controller_output['fwd_res']['lat_signature'].keys()
                warnings.warn(message='Phenotype selection ot specified, using: ' + str(selected_pheno))
            for cur_signature in selected_signature:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['lat_signature'][cur_signature].numpy()).set_axis(
                        [cur_signature + '.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['lat_signature'][cur_signature].shape[1]
                        )], axis=1, inplace=False)
                )

        # Signature outputs
        if dump_signature_out:
            if selected_signature is None:
                selected_pheno = controller_output['fwd_res']['signature_out'].keys()
                warnings.warn(message='Phenotype selection ot specified, using: ' + str(selected_pheno))
            for cur_signature in selected_signature:
                lat_all.append(
                    pd.DataFrame(controller_output['fwd_res']['signature_out'][cur_signature].numpy()).set_axis(
                        [cur_signature + '.out.' + str(i + 1) for i in range(
                            controller_output['fwd_res']['signature_out'][cur_signature].shape[1]
                        )], axis=1, inplace=False)
                )

        # Assemble dataframe (and set cell names)
        if rownames is not None:
            lat_all = pd.concat(lat_all, axis=1).set_index(rownames)

        # Save CSV
        if compression == 'hdf':
            lat_all.to_hdf(path, 'lat_all', mode='w')
        elif compression != 'none':
            lat_all.to_csv(path, compression=compression)
        else:
            lat_all.to_csv(path)

    def save_config(self, config_dict, json_path):
        """
        Persist pipeline configuration to JSON.

        :param config_dict: Complete experiment configuration
        :type config_dict: dict
        :param json_path: Output path for JSON file
        :type json_path: str

        :return: None
        """
        with open(json_path, 'w') as f:
            json.dump(config_dict, f)

    def save_splits(self, split_dict, pkl_path):
        """
        Serialize data splits to pickle format.

        :param split_dict: Data partition dictionary containing all, overall_train/test, Phenotype train/test identifiers
        :type split_dict: dict
        :param pkl_path: Output path for pickle file
        :type pkl_path: str

        :return: None
        """
        with open(pkl_path, 'wb') as f:
            pickle.dump(split_dict, f)