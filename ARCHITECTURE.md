# Repository Architecture

## Project Structure

```
├── .github                   <- GitHub Actions workflows
│
├── baseline_models           <- Neural network implementations and training scripts
│   │                            for 6 model architectures across 5 training configurations
│   ├── convnext                 <- ConvNeXt model
│   ├── encdec_lstm              <- Encoder-Decoder LSTM
│   ├── pao_model                <- PAO (Pritchard et al.) model
│   ├── pure_resLSTM             <- Residual LSTM
│   ├── squeezeformer            <- Squeezeformer architecture
│   └── unet                     <- U-Net architecture
│       │
│       └── training_*/          <- Training configurations for each model:
│           ├── training_conf_loss      <- Confidence loss configuration
│           ├── training_default        <- Standard configuration
│           ├── training_diff_loss      <- Difference loss configuration
│           ├── training_multirep       <- Multi-representation configuration
│           └── training_v6             <- Hu et al. 2025 configuration
│               │
│               ├── conf/               <- Hydra configuration files (seed variations)
│               ├── slurm/              <- SLURM job submission scripts
│               ├── *.py                <- Model architecture and training scripts
│               └── wrap_model.py       <- Model wrapper for online inference
│
├── climsim_utils             <- Installable Python package for data preprocessing
│   └── data_utils.py            and model evaluation utilities
│
├── dataset_statistics        <- Precomputed statistics of input and output variables
│   ├── input2D/                 <- 2D input variable statistics
│   ├── input3D/                 <- 3D input variable statistics (per level)
│   ├── output2D/                <- 2D output variable statistics
│   └── output3D/                <- 3D output variable statistics (per level)
│
├── demo_notebooks            <- Example Jupyter notebooks
│   ├── quickstart_example.ipynb <- Getting started with the dataset
│   └── water_conservation.ipynb <- Water conservation analysis
│
├── evaluation                <- Model evaluation scripts and notebooks
│   ├── offline/                 <- Offline evaluation (test set metrics)
│   │   ├── *.py                    - Scripts for computing metrics
│   │   └── *.ipynb                 - Visualization notebooks
│   │
│   └── online/                  <- Online evaluation (coupled simulation metrics)
│       ├── *.py                    - Scripts for extracting simulation data
│       └── *.ipynb                 - Analysis and visualization notebooks
│
├── figures                   <- Publication figures and analysis plots
│   └── climsim_figures.ipynb    <- Additional figure generation notebook
│
├── grid_info                 <- Grid information and mapping files
│   ├── ClimSim_low-res_grid-info.nc           <- Grid metadata
│   ├── map_ne4pg2_to_180x360_lowres.nc        <- Regridding weights (7.4MB)
│   └── map_ne30pg2_to_180x360_highres.nc      <- Regridding weights (11MB)
│
├── mmf_scripts               <- Multi-scale Modeling Framework (E3SM-MMF) utilities
│   ├── longeval_gpu.py          <- Long evaluation on GPU
│   ├── speedeval_cpu.py         <- Speed benchmarking on CPU
│   └── speedeval_gpu.py         <- Speed benchmarking on GPU
│
├── online_ensembling         <- Scripts for running online ensemble simulations
│   ├── conf/                    <- Configuration-specific ensembles
│   ├── v2_rh_mc/                <- v2 relative humidity + microphysics ensembles
│   └── v6/                      <- Hu et al. 2025 configuration ensembles
│
├── preprocessing             <- Data preprocessing scripts and examples
│   ├── code_examples/           <- Example notebooks for preprocessing tasks
│   │   ├── dataset_creation/       - Creating datasets with different variables
│   │   ├── feature_expansion/      - Adding new input features
│   │   └── transformation/         - Input/output scaling and transformations
│   │
│   ├── data_prep_scripts/       <- Production data preparation scripts
│   │   ├── v2_kaggle/              - Kaggle competition dataset
│   │   ├── v2_rh_mc/               - v2 with RH and microphysics
│   │   ├── v6/                     - Hu et al. 2025 configuration
│   │   └── v7/                     - v7 configuration
│   │
│   └── normalizations/          <- Precomputed normalization factors
│       ├── inputs/                 - Input variable normalizations
│       └── outputs/                - Output variable normalizations
│
├── tests                     <- Unit tests
│   ├── conftest.py              <- pytest configuration
│   ├── test_data_utils.py       <- Tests for data utilities
│   ├── testing_data_utils_with_backends.py
│   └── unit_tests.ipynb         <- Interactive testing notebook
│
├── website                   <- Documentation website
│
├── preprocess_figure_data.ipynb  <- Compute expensive metrics and save results
│                                    (preprocessing step for paper figures)
│
├── generate_paper_figures.ipynb  <- Load precomputed data and generate all
│                                    main paper + supplementary figures
│
├── .gitignore                <- Files ignored by git
├── LICENSE                   <- Project license
├── README.md                 <- Project overview and getting started guide
├── ARCHITECTURE.md           <- This file - repository structure documentation
└── setup.py                  <- Package installation script for climsim_utils
```

## Training Configuration Details

Each of the 6 models can be trained with 5 different configurations:

1. **training_default**: Standard training with baseline loss function
2. **training_conf_loss**: Training with confidence loss for uncertainty quantification
3. **training_diff_loss**: Training with difference loss
4. **training_multirep**: Multi-representation inputs (additional derived variables)
5. **training_v6**: Hu et al. 2025 configuration with expanded variable list

Each configuration supports multiple random seeds (7, 43, 1024) for ensemble training.

## Key Files

- **Model Architecture**: `baseline_models/{model}/training_{config}/{model}.py`
- **Training Script**: `baseline_models/{model}/training_{config}/train_{model}.py`
- **Model Wrapper**: `baseline_models/{model}/training_{config}/wrap_model.py`
- **Configuration**: `baseline_models/{model}/training_{config}/conf/config.yaml`
- **Data Utilities**: `climsim_utils/data_utils.py`

## Workflow

1. **Data Preparation**: Use scripts in `preprocessing/data_prep_scripts/`
2. **Model Training**: Submit SLURM jobs from `baseline_models/*/training_*/slurm/`
3. **Offline Evaluation**: Run scripts in `evaluation/offline/`
4. **Online Evaluation**: Run ensemble simulations with `online_ensembling/`
5. **Figure Generation**:
   - Run `preprocess_figure_data.ipynb` to compute metrics
   - Run `generate_paper_figures.ipynb` to create visualizations

## Evaluation Pipeline

The evaluation workflow consists of multiple phases that precompute expensive metrics for efficient figure generation.

### Phase 1: Offline Inference (evaluation/offline/)

**Primary Script**: `offline_inference_test.py`

This script performs inference on all trained models against the test set and computes R² scores.

**Inputs:**
- Trained model checkpoints for 6 models × 5 configurations × 3 seeds (90 combinations)
- Test set input/target data from preprocessing
- Normalization files

**Outputs:**
- **Prediction files** (`.npz`): Model predictions for each configuration/model combination
  - Location: `/pscratch/.../test_preds/{config}/{config}_{model}_preds.npz`
  - Format: Keys `seed_7`, `seed_43`, `seed_1024` containing prediction arrays
  - Example: `standard_unet_preds.npz` with shape `(time, latlon, features)`

- **R² score files** (`.pkl`): Precomputed R² scores for each model
  - Location: `/pscratch/.../test_preds/{config}/{config}_{model}_r2.pkl`
  - Used by notebooks to avoid recomputing expensive R² calculations

**Purpose**: This is the most computationally expensive step - running inference on 90 model combinations and computing per-feature R² scores across the entire test set.

### Phase 2: Offline Figure Generation (evaluation/offline/)

**Scripts:**
- `create_offline_binned_moistening_bias_config_comparison.py`
- `create_offline_binned_moistening_bias_model_comparison.py`
- `create_offline_zonal_mean_tendency_bias.py`
- `create_offline_zonal_mean_tendency_conf.py`
- `get_offline_bias_minmax.py`

**Inputs:**
- Prediction `.npz` files from Phase 1
- Test set targets

**Outputs:**
- PNG figures showing offline metrics (bias profiles, zonal means, etc.)
- Saved to `/pscratch/.../climsim3_figures/offline/`

**Purpose**: These scripts load precomputed predictions and generate diagnostic plots. They do not save intermediate processed data - they are pure visualization utilities.

### Phase 3: Online Data Preprocessing (preprocess_figure_data.ipynb)

**Inputs:**
- R² score `.pkl` files from Phase 1
- Online simulation output (NetCDF files from E3SM-MMF coupled runs)
  - Location: `/pscratch/.../online_runs/climsim3_ensembles_good/`
  - Multi-year (4-year and 5-year) simulation data

**Processing:**
- Loads multi-year online simulation datasets
- Computes expensive statistics:
  - RMSE calculations across spatial/temporal dimensions
  - Hourly precipitation statistics
  - Multi-year averaged fields
- Creates comparison datasets for all model/config combinations

**Outputs** (saved as `.pkl` files):
- `ds_nn_{config}_{years}_year.pkl` - Online simulation datasets (xarray)
- `ds_nn_rmse_{config}_{years}_year.pkl` - RMSE statistics
- `nn_hourly_prect_{config}_{years}_year.pkl` - Hourly precipitation data
- `mmf_{ref}_rmse_{years}_year.pkl` - MMF reference simulation metrics

**Purpose**: This notebook performs the expensive I/O and computation on large multi-year simulation datasets, saving the processed results for quick loading during figure generation.

### Phase 4: Paper Figure Generation (generate_paper_figures.ipynb)

**Inputs:**
- All `.pkl` files from Phase 3
- Some prediction files from Phase 1 (for specific analyses)

**Outputs:**
- All main text figures
- All supplementary figures
- Saved as high-resolution PNGs/PDFs

**Purpose**: Loads precomputed data and generates publication-quality figures. Runs quickly since all expensive computations were done in Phases 1 and 3.

### Summary of Data Flow

```
Phase 1: offline_inference_test.py
    ↓ (saves .npz predictions + .pkl R² scores)

Phase 2: create_offline_*.py
    ↓ (loads .npz, generates offline diagnostic figures)

Phase 3: preprocess_figure_data.ipynb
    ↓ (loads R² .pkl + online NetCDF → saves processed .pkl)

Phase 4: generate_paper_figures.ipynb
    ↓ (loads all .pkl → generates paper figures)
```

**Key Insight**: The precomputing workflow separates expensive operations (model inference, multi-year simulation loading) from figure generation, enabling rapid iteration on visualizations without rerunning computationally expensive steps.
