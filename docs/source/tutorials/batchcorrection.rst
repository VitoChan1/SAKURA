Batch Effect Handling in SAKURA
===============================

Batch effects are technical variations that occur between different experimental batches,
which can confound biological signals in single-cell RNA sequencing data.
SAKURA utilizes flexible approaches to handle batch effects,
allowing users to choose the strategy that best fits their data and analysis goals.

Approach 1: Pre-corrected Expression Input
------------------------------------------

Perform batch correction methods on the raw expression matrix prior to utilizing SAKURA.
The resulting expression matrix will serve as the input dataset for implementing SAKURA.

:**Advantages**:
  - Corrects at the expression level, preserving biological signals
  - Flexible choice of established methods with extensive validation
  - SAKURA operates normally without modifications

:**Considerations**:
  - May over-correct and remove subtle biological variations
  - Requires careful parameter tuning for optimal results due to transformed statistical properties \
    and potential information loss in the batch corrected data

:**Suitable Methods**:
  - `Seurat CCA/RPCA <https://satijalab.org/seurat>`_: Batch effect correction aligning canonical basis vectors or reciprocal PCA
  - `scVI <https://scvi-tools.org>`_: Probabilistic modeling of batch effects
  - `Liger <https://github.com/welch-lab/liger>`_: Batch effect correction relies on integrative non-negative matrix factorization

Approach 2: Post-hoc Embedding Correction
-----------------------------------------

Use SAKURA result embeddings as input to external batch correction methods,
and then specifically remove technical batch variation within this low-dimensional space
already enriched for biological signal.

.. note::
    Use SAKURA embeddings as a **functional substitute** for PCA coordinates
    in batch correction, but ensure the method does not **mathematically depends on PCA-specific constructs**
    (e.g., gene loadings for matrix reconstruction).

:**Typical Workflow**:
  1. **Extract batch information** from metadata during data preprocessing (``batch``, ``donor``, ``sequencing_run``, etc.)
  2. **Generate SAKURA embeddings** using standard training pipeline
  3. **Apply batch correction** to SAKURA embeddings using external tools
  4. **Use corrected embeddings** for clustering, visualization, and analysis

:**Advantages**:
  - Preserves SAKURA's biological signal learning
  - Reduces correction computational cost and time with low-dimensional SAKURA embedding
  - Enables modular evaluation on both SAKURA feature learning quality and batch correction efficacy

:**Considerations**:
  - Requires compatible correction methods applied after feature learning
  - May involve iterative optimization between external correction methods and SAKURA \
    targeting two objectives, i.e. biological signal learning and batch effect correction

:**Suitable Methods**:
  - `Harmony <https://github.com/immunogenomics/harmony>`_: Integration using diversity clustering correction
  - `fastMNN <https://github.com/MarioniLab/FurtherMNN2018/>`_: Batch effect correction aligning mutual nearest neighbors by cosine distance


Approach 3: Knowledge-Guided Training with Pre-Corrected Embeddings
-------------------------------------------------------------------

Use pre-computed, batch-corrected low dimensional embeddings as the knowledge input for SAKURA's feature training.

.. note::
    This option is designed for scenarios where a reliable, batch-corrected
    low-dimensional representation of the data already exists or is preferred to be
    generated upstream.The core idea is to **format prior knowledge** about the desired,
    batch-effect-free cell-state together with necessary cell and feature metadata
    as the knowledge input to SAKURA for feature learning.

**Configuration Example**:

.. code-block:: json

    {
        "dataset": {
            ...
            "pheno_csv_path": "./<dataset>/pheno_df_with_batch_effect_correction_features.csv",
            "pheno_meta_path": "./<dataset>_<signature/phenotype>_bec/pheno_meta.json",
            "selected_pheno": [
                "BE_cor",
                ...
            ],
            ...
        },
        ...
        "story": [
          {
            ...
            "<pheno_with_>batch_effect_correction": {
                "use_split": "overall_train",
                "batch_size": 100,
                "train_main_latent": "False",
                "train_pheno": "True",
                "train_signature": "False",
                "selected_pheno": {
                    ...
                    "BE_cor": {
                        "loss": "*",
                        "regularization": "*"
                    }
                }
            },
          }
        ]
    }

:**Advantages**:
  - Separates the complex problem of batch correction from knowledge-guided deep learning
  - Flexible integration with existing workflows where batch correction is a standardized upstream step

:**Considerations**:
  - Requires appropriate choice of upstream batch correction methods and decent validation
  - Requires careful parameter tuning balancing the intensity of batch correction and other biological signal learning
  - Evaluation is holistic; difficult to disentangle the combined effect due to the fused learning.

:**Suitable Methods**:
  - `Harmony <https://github.com/immunogenomics/harmony>`_: Integration using diversity clustering correction
  - `BBKNN <https://github.com/Teichlab/bbknn>`_: Batch balanced k-nearest neighbors
  - `Scanorama <https://github.com/brianhie/scanorama>`_: Panoramic stitching of datasets via mutual nearest neighbors


Conclusion
----------

SAKURA's flexible architecture supports multiple strategies for batch effect handling.
The choice between pre-correction, post-hoc correction, or corrected embedding knowledge input
depends on the severity of batch effects, data complexity, and specific research questions.

For most applications, we recommend starting with **Approach 1** (pre-corrected expression)
for well-established datasets, or **Approach 2** (post-hoc embedding correction) when SAKURA's
biological feature extraction is prioritized.
