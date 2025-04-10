"""
Multi-component model architecture assembler
"""

import torch

import sakura.models.modules as model


class Extractor(torch.nn.Module):
    """
    End-to-end multi-component model architecture assembler and forward orchestrator

    Despite the name 'Extractor', this class acts as the **central hub** that integrates
    all submodules of SAKURA based on configurations. The name conceptually emphasis the
    class role in extracting high-level representations through the assembled components
    as dimensionality reduction being the main task.

    :param input_dim: The dimensionality of the model inputs
    :type input_dim: int
    :param signature_config: Model configuration settings
        for the signature regression branch
    :type signature_config: dict[str, Any], optional
    :param pheno_config: Model configuration settings for the phenotype prediction/regression branch
    :type pheno_config: dict[str, Any], optional
    :param main_lat_config: Model configuration settings for the main latent representation
        of the autoencoder backbone
    :type main_lat_config: dict[str, Any]
    :param pre_encoder_config: Model configuration settings for the pre-encoder stage
    :type pre_encoder_config: dict[str, Any], optional
    :param verbose: Whether to enable verbose console logging, defaults to False
    :type verbose: bool

    Architecture Composition:
        • pre_encoder (nn.Module): Raw input preprocessing/initial feature transformation
        • main_latent_compressor (nn.Module): Core bottleneck for dimensionality reduction
        • signature_latent_compressors (nn.ModuleDict): Task-specific latent extraction branch for signature analysis
        • signature_regressors (nn.ModuleDict): Signature regression head
        • pheno_latent_compressors (nn.ModuleDict): ask-specific latent extraction branch for phenotype analysis
        • pheno_models (nn.ModuleDict): Phenotype prediction/regression head
        • decoder (nn.Module): Reconstruction/upsampling component

    Forward Flow:
        1. <Input> → pre_encoder → <Pre-latent> → main_latent_compressor → <Main latent>
        2. <Main latent> OR <Pre-latent> → parallel signature/pheno processing branches → parallel signature regressors and pheno_models
        3. Main latent → decoder → final outputs
    """

    def __init__(self,
                 input_dim: int,
                 signature_config=None, pheno_config=None, main_lat_config=None,
                 pre_encoder_config=None, verbose=False):
        super(Extractor, self).__init__()

        # Verbose logging for debugging
        self.verbose = verbose

        self.input_dim = input_dim
        self.main_lat_config = main_lat_config

        # Legacy model structure parameters (back compatibility)
        self.encoder_neurons = self.main_lat_config.get('encoder_neurons')  # Actually the dimension of pre-encoder outputs
        self.decoder_neurons = self.main_lat_config.get('decoder_neurons')
        self.main_latent_dim = self.main_lat_config.get('latent_dim')

        self.signature_config = signature_config
        self.pheno_config = pheno_config
        self.pre_encoder_config = pre_encoder_config

        # Pre-encoder
        # Back compatibility: if there is no pre-encoder config, use default pre-encoder arch (in --> h --> h)
        if self.pre_encoder_config is None:
            pre_encoder = model.FCPreEncoder(input_dim=self.input_dim,
                                             output_dim=self.encoder_neurons,
                                             hidden_neurons=self.encoder_neurons)
        else:
            # When pre-encoder is customized, self.encoder_neurons should be overriden by the output of pre-encoder
            self.encoder_neurons = self.pre_encoder_config.get('pre_encoder_out_dim', self.encoder_neurons)
            pre_encoder = model.FCPreEncoder(input_dim=self.input_dim,
                                             output_dim=self.encoder_neurons,
                                             hidden_neurons=self.pre_encoder_config.get('hidden_neurons'),
                                             hidden_layers=self.pre_encoder_config.get('hidden_layers'),
                                             dropout=(self.pre_encoder_config.get('dropout') == 'True'),
                                             dropout_input=(self.pre_encoder_config.get('dropout_input') == 'True'),
                                             dropout_input_p=self.pre_encoder_config.get('dropout_input_p', 0.5),
                                             dropout_hidden=(self.pre_encoder_config.get('dropout_hidden') == 'True'),
                                             dropout_hidden_p=self.pre_encoder_config.get('dropout_hidden_p', 0.5))
        # Main Latent Compressor
        if main_lat_config.get('encoder_config') is None:
            # Back compatibility (legacy mode)
            main_latent_compressor = model.FCCompressor(input_dim=self.encoder_neurons,
                                                        output_dim=self.main_latent_dim)
        else:
            main_latent_compressor = model.FCCompressor(input_dim=self.encoder_neurons,
                                                        output_dim=self.main_latent_dim,
                                                        hidden_neurons=main_lat_config['encoder_config'].get('hidden_neurons'),
                                                        hidden_layers=main_lat_config['encoder_config'].get('hidden_layers'),
                                                        dropout=(main_lat_config['encoder_config'].get('dropout') == 'True'),
                                                        dropout_input=(main_lat_config['encoder_config'].get('dropout_input') == 'True'),
                                                        dropout_input_p=main_lat_config['encoder_config'].get('dropout_input_p',0.5),
                                                        dropout_hidden=(main_lat_config['encoder_config'].get('dropout_hidden') == 'True'),
                                                        dropout_hidden_p=main_lat_config['encoder_config'].get('dropout_hidden_p', 0.5))

        # Signature Latent Compressor
        total_latent_dim = self.main_latent_dim
        signature_latent_compressors = torch.nn.ModuleDict()
        if self.signature_config is not None:
            for cur_signature in self.signature_config.keys():
                model_details = self.signature_config[cur_signature].get('model')
                if model_details is None or model_details.get('attach') != 'True':
                    # Create extra dimensions by default
                    total_latent_dim = total_latent_dim + self.signature_config[cur_signature]['signature_lat_dim']

                    encoder_config = self.signature_config[cur_signature].get('encoder_config')
                    if encoder_config is None:
                        # Default: a linear compressor
                        signature_latent_compressors[cur_signature] = model.FCCompressor(input_dim=self.encoder_neurons,
                                                                                         output_dim=self.signature_config[cur_signature]['signature_lat_dim'])
                    else:
                        signature_latent_compressors[cur_signature] = model.FCCompressor(input_dim=self.encoder_neurons,
                                                                                         output_dim=self.signature_config[cur_signature]['signature_lat_dim'],
                                                                                         hidden_neurons=encoder_config.get('hidden_neurons'),
                                                                                         hidden_layers=encoder_config.get('hidden_layers'))

        # Signature regressor
        signature_regressors = torch.nn.ModuleDict()
        if self.signature_config is not None:
            for cur_signature in self.signature_config.keys():
                model_details = self.signature_config[cur_signature].get('model')
                if self.verbose:
                    print("Building phenotype module: ", cur_signature)
                    print("Model details: ", model_details)
                if model_details is None:
                    # Legacy: by default, use a linear regressor
                    signature_regressors[cur_signature] = model.LinRegressor(
                        input_dim=self.signature_config[cur_signature]['signature_lat_dim'],
                        output_dim=self.signature_config[cur_signature]['signature_out_dim'],
                        output_activation_function=model_details.get('output_activation_function', 'identity')
                    )
                else:
                    if model_details['type'] == 'FCRegressor':
                        signature_regressors[cur_signature] = model.FCRegressor(
                            input_dim=self.signature_config[cur_signature]['signature_lat_dim'],
                            output_dim=self.signature_config[cur_signature]['signature_out_dim'],
                            hidden_neurons=model_details.get('hidden_neurons', 5),
                            hidden_layers=model_details.get('hidden_layers'),
                            output_activation_function=model_details.get('output_activation_function', 'identity'),
                            dropout=(model_details.get('dropout') == 'True'),
                            dropout_input=(model_details.get('dropout_input') == 'True'),
                            dropout_input_p=model_details.get('dropout_input_p', 0.5),
                            dropout_hidden=(model_details.get('dropout_hidden') == 'True'),
                            dropout_hidden_p=model_details.get('dropout_hidden_p', 0.5)
                        )
                    elif model_details['type'] == 'LinRegressor':
                        signature_regressors[cur_signature] = model.LinRegressor(
                            input_dim=self.signature_config[cur_signature]['signature_lat_dim'],
                            output_dim=self.signature_config[cur_signature]['signature_out_dim'],
                            output_activation_function=model_details.get('output_activation_function', 'identity')
                        )
                    elif model_details['type'] == 'bypass' or model_details['type'] == 'Identity':
                        signature_regressors[cur_signature] = torch.nn.Identity()
                    else:
                        raise ValueError('Unsupported phenotype side task model')

        # Phenotype Latent Compressor
        pheno_latent_compressors = torch.nn.ModuleDict()
        if self.pheno_config is not None:
            for cur_pheno in self.pheno_config.keys():
                model_details = self.pheno_config[cur_pheno].get('model')
                if model_details is None or model_details.get('attach') != 'True':
                    total_latent_dim = total_latent_dim + self.pheno_config[cur_pheno]['pheno_lat_dim']

                    encoder_config = self.pheno_config[cur_pheno].get('encoder_config')
                    if encoder_config is None:
                        # Default: linear compressor
                        pheno_latent_compressors[cur_pheno] = model.FCCompressor(input_dim=self.encoder_neurons,
                                                                                 output_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'])
                    else:
                        pheno_latent_compressors[cur_pheno] = model.FCCompressor(input_dim=self.encoder_neurons,
                                                                                 output_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                                                                                 hidden_neurons=encoder_config.get('hidden_neurons'),
                                                                                 hidden_layers=encoder_config.get('hidden_layers'))

        # Category classifier(/regressor)
        pheno_models = torch.nn.ModuleDict()
        if self.pheno_config is not None:
            for cur_pheno in self.pheno_config.keys():
                model_details = self.pheno_config[cur_pheno].get('model')
                if self.verbose:
                    print("Building phenotype module: ", cur_pheno)
                    print("Model details: ", model_details)
                if model_details is None:
                    # Legacy: by default, use a linear classifier
                    pheno_models[cur_pheno] = model.LinClassifier(
                        input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                        output_dim=self.pheno_config[cur_pheno]['pheno_out_dim']
                    )
                else:
                    if model_details['type'] == "LinClassifier":
                        pheno_models[cur_pheno] = model.LinClassifier(
                            input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                            output_dim=self.pheno_config[cur_pheno]['pheno_out_dim']
                        )
                    elif model_details['type'] == 'FCClassifier':
                        pheno_models[cur_pheno] = model.FCClassifier(
                            input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                            output_dim=self.pheno_config[cur_pheno]['pheno_out_dim'],
                            hidden_neurons=model_details.get('hidden_neurons', 5),
                            hidden_layers=model_details.get('hidden_layers'),
                            dropout=(model_details.get('dropout') == 'True'),
                            dropout_input=(model_details.get('dropout_input') == 'True'),
                            dropout_input_p=model_details.get('dropout_input_p', 0.5),
                            dropout_hidden=(model_details.get('dropout_hidden') == 'True'),
                            dropout_hidden_p=model_details.get('dropout_hidden_p', 0.5),
                        )
                    elif model_details['type'] == 'FCRegressor':
                        pheno_models[cur_pheno] = model.FCRegressor(
                            input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                            output_dim=self.pheno_config[cur_pheno]['pheno_out_dim'],
                            hidden_neurons=model_details.get('hidden_neurons', 5),
                            hidden_layers=model_details.get('hidden_layers'),
                            dropout=(model_details.get('dropout') == 'True'),
                            dropout_input=(model_details.get('dropout_input') == 'True'),
                            dropout_input_p=model_details.get('dropout_input_p', 0.5),
                            dropout_hidden=(model_details.get('dropout_hidden') == 'True'),
                            dropout_hidden_p=model_details.get('dropout_hidden_p', 0.5)
                        )
                    elif model_details['type'] == 'LinRegressor':
                        pheno_models[cur_pheno] = model.LinRegressor(
                            input_dim=self.pheno_config[cur_pheno]['pheno_lat_dim'],
                            output_dim=self.pheno_config[cur_pheno]['pheno_out_dim']
                        )
                    elif model_details['type'] == 'bypass' or model_details['type'] == 'Identity':
                        pheno_models[cur_pheno] = torch.nn.Identity()
                    else:
                        raise ValueError('Unsupported phenotype side task model')

        # Decoder configuration
        if main_lat_config.get('decoder_config') is None:
            # Legacy
            decoder = model.FCDecoder(input_dim=total_latent_dim,
                                      output_dim=self.input_dim,
                                      hidden_neurons=self.decoder_neurons)
        else:
            decoder = model.FCDecoder(input_dim=total_latent_dim,
                                      output_dim=self.input_dim,
                                      hidden_neurons=main_lat_config['decoder_config'].get('hidden_neurons'),
                                      hidden_layers=main_lat_config['decoder_config'].get('hidden_layers'),
                                      output_activation_function=main_lat_config['decoder_config'].get('output_activation_function'),
                                      dropout=(main_lat_config['decoder_config'].get('dropout') == 'True'),
                                      dropout_input=(main_lat_config['decoder_config'].get('dropout_input') == 'True'),
                                      dropout_input_p=main_lat_config['decoder_config'].get('dropout_input_p', 0.5),
                                      dropout_hidden=(main_lat_config['decoder_config'].get('dropout_hidden') == 'True'),
                                      dropout_hidden_p=main_lat_config['decoder_config'].get('dropout_hidden_p', 0.5))

        # Assemble model
        self.model = torch.nn.ModuleDict({
            'pre_encoder': pre_encoder,
            'main_latent_compressor': main_latent_compressor,
            'signature_latent_compressors': signature_latent_compressors,
            'signature_regressors': signature_regressors,
            'pheno_latent_compressors': pheno_latent_compressors,
            'pheno_models': pheno_models,
            'decoder': decoder
        })

        if self.verbose:
            print("Model built:")
            print(self.model)

    def forward(self, batch,
                forward_signature=True, selected_signature=None,
                forward_pheno=True, selected_pheno=None,
                forward_main_latent=True, forward_reconstruction=True,
                detach=False, detach_from=''):
        """
        Forward extractor framework with control over computation branches

        Orchestrates data flow through the assembled modular architecture, enabling selective
        activation of task branches and gradient flow control.

        :param batch: Gene expression tensors, shape should be (N,M), where N is number of cell,
            M is number of gene
        :type batch: torch.Tensor
        :param forward_signature: Whether to forward signature supervision part, defaults to True
        :type forward_signature: bool
        :param selected_signature: List of selected signatures to be forwarded,
            None to forward all signatures
        :type selected_signature: list[str], optional
        :param forward_pheno: Whether to forward phenotype supervision part, defaults to True
        :type forward_pheno: bool
        :param selected_pheno: List of selected phenotypes to be forwarded,
            None to forward all phenotypes
        :type selected_pheno: list[str], optional
        :param forward_main_latent: Whether to forward main latent part, defaults to True
        :type forward_main_latent: bool
        :param forward_reconstruction*: Whether to forward decoder reconstruction part, defaults to True
        :type forward_reconstruction: bool
        :param detach: Should the gradient be blocked from midway of the network
            as specified in <detach_from>, defaults to False
        :type detach: bool
        :param detach_from: Specific component from which the gradient should be blocked
            if <detach> is True
        :type detach_from: Literal['pre_encoder', 'encoder'] or str

        .. note::
            **<forward_reconstruction>:** The decoder reconstruction part could only be forwarded when
            all latent dimensions are forwarded.

            <detach_from> options:
            - 'pre_encoder' (lat_pre will be detached, pre_encoder will not be trained);
            - 'encoder' (main_lat, pheno_lat, signature_lat will be detached, neither pre-encoder nor encoder will be trained).

            **Gradient reverse layer** and **gradient neutralize layer** related computations are done
            in :mod:`model_controllers.extractor_controller`.

        :return: a dictionary containing hierarchical outputs with keys of model forwarding
        :rtype: dict[str, torch.Tensor]
        """
        # Forward Pre Encoder
        lat_pre = self.model['pre_encoder'](batch)

        # Handle detach from pre_encoder
        if detach and detach_from == 'pre_encoder':
            lat_pre = lat_pre.detach()

        # Forward Main Latent
        lat_main = torch.Tensor()
        lat_all = torch.Tensor()
        if forward_main_latent:
            lat_main = self.model['main_latent_compressor'](lat_pre)

            # Handle detach from encoder
            if detach and detach_from == 'encoder':
                lat_main = lat_main.detach()

            lat_all = lat_main

        # Forward signature supervision
        lat_signature = dict()
        signature_out = dict()
        attached_signatures = list()
        if forward_reconstruction or forward_signature:
            if selected_signature is None:
                # Select all signature
                selected_signature = self.signature_config.keys()
            for cur_signature in selected_signature:
                model_details = self.signature_config[cur_signature].get('model')

                # Check if current signature requires handling of attachments
                if model_details is not None:
                    if model_details.get('attach') == 'True':
                        attached_signatures.append(cur_signature)
                        continue

                lat_signature[cur_signature] = self.model['signature_latent_compressors'][cur_signature](lat_pre)

                # Handle detach from encoder
                if detach and detach_from == 'encoder':
                    lat_signature[cur_signature] = lat_signature[cur_signature].detach()

                lat_all = torch.cat((lat_all, lat_signature[cur_signature]), 1)
                signature_out[cur_signature] = self.model['signature_regressors'][cur_signature](
                    lat_signature[cur_signature])

        # Forward phenotype supervision
        lat_pheno = dict()
        pheno_out = dict()
        attached_phenos = list()
        if forward_reconstruction or forward_pheno:
            if selected_pheno is None:
                selected_pheno = self.pheno_config.keys()
            for cur_pheno in selected_pheno:
                model_details = self.pheno_config[cur_pheno].get('model')
                if model_details is not None:
                    if model_details.get('attach') == 'True':
                        attached_phenos.append(cur_pheno)
                        continue
                lat_pheno[cur_pheno] = self.model['pheno_latent_compressors'][cur_pheno](lat_pre)

                # Handle detach from encoder
                if detach and detach_from == 'encoder':
                    lat_pheno[cur_pheno] = lat_pheno[cur_pheno].detach()

                lat_all = torch.cat((lat_all, lat_pheno[cur_pheno]), 1)
                pheno_out[cur_pheno] = self.model['pheno_models'][cur_pheno](lat_pheno[cur_pheno])

        # Reconstruct input gene expression profiles
        re_x = None
        if forward_reconstruction:
            re_x = self.model['decoder'](lat_all)

        # Process attached signature/pheno tasks
        def handle_attach(model_details):
            if model_details['attach_to'] == 'main_lat':
                return lat_main
            elif model_details['attach_to'] == 'signature_lat':
                return lat_signature[model_details['attach_key']]
            elif model_details['attach_to'] == 'pheno_lat':
                return lat_pheno[model_details['attach_key']]
            elif model_details['attach_to'] == 'all_lat':
                return lat_all
            elif model_details['attach_to'] == 'multiple':
                lat_cur = torch.Tensor()
                for cur_attach in model_details['attach_key']:
                    if cur_attach['type'] == 'pheno':
                        lat_cur = torch.cat((lat_cur, lat_pheno[cur_attach['key']]), 1)
                    elif cur_attach['type'] == 'signature':
                        lat_cur = torch.cat((lat_cur, lat_signature[cur_attach['key']]), 1)
                    elif cur_attach['type'] == 'main':
                        lat_cur = torch.cat((lat_cur, lat_main), 1)
                return lat_cur
            else:
                raise NotImplementedError("Unsupported type of attach.")

        # Attached signatures
        if forward_signature:
            for cur_signature in attached_signatures:
                model_details = self.signature_config[cur_signature].get('model')
                lat_signature[cur_signature] = handle_attach(model_details)
                signature_out[cur_signature] = self.model['signature_regressors'][cur_signature](lat_signature[cur_signature])

        # Attached phenos
        if forward_pheno:
            for cur_pheno in attached_phenos:
                model_details = self.pheno_config[cur_pheno].get('model')
                lat_pheno[cur_pheno] = handle_attach(model_details)
                pheno_out[cur_pheno] = self.model['pheno_models'][cur_pheno](lat_pheno[cur_pheno])

        return {
            'x': batch,
            'lat_pre': lat_pre,
            'lat_main': lat_main,
            'lat_signature': lat_signature,
            'signature_out': signature_out,
            'lat_pheno': lat_pheno,
            'pheno_out': pheno_out,
            're_x': re_x,
            'lat_all': lat_all
        }
