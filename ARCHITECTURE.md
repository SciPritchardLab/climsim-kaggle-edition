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
