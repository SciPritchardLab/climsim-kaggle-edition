# ClimSim Kaggle Edition: Online Testing of Competition-Winning Architectures

This repository accompanies a forthcoming paper that evaluates neural network architectures from the 2024 LEAP ClimSim Kaggle competition in "online" coupled climate simulations with E3SM-MMF (Energy Exascale Earth System Model - Multi-scale Modeling Framework).

## Overview

The [ClimSim Kaggle competition](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim) challenged participants to develop machine learning emulators of cloud and convection processes for climate modeling. This repository tests whether architectures that performed well in offline metrics also produce stable, physically realistic results when coupled to a climate model.

### Key Features

- **6 Model Architectures**: Implementations of winning Kaggle competition architectures plus baseline
- **5 Training Configurations**: Architecture-agnostic design variations inspired by competition insights
- **Ensemble Training**: Multiple random seeds (7, 43, 1024) for robust evaluation
- **Online Testing Framework**: Uses [FTorch-based E3SM-MMF](https://github.com/zyhu-hu/E3SM_nvlab/tree/ftorch/climsim_scripts/perlmutter_scripts) for coupled simulations
- **Comprehensive Evaluation**: Offline metrics, online simulation analysis, and figure generation scripts

## Repository Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed structure documentation.

```
├── baseline_models/          # Model implementations and training scripts
│   ├── convnext/             # ConvNeXt architecture
│   ├── encdec_lstm/          # Encoder-Decoder LSTM
│   ├── pao_model/            # PAO model (3rd place)
│   ├── pure_resLSTM/         # Pure ResLSTM (2nd place)
│   ├── squeezeformer/        # Squeezeformer (1st place)
│   └── unet/                 # U-Net baseline
│       └── training_*/       # 5 training configurations per model
│
├── evaluation/               # Evaluation scripts and notebooks
│   ├── offline/              # Test set metrics
│   └── online/               # Coupled simulation analysis
│
├── preprocessing/            # Data preparation scripts
├── online_ensembling/        # Online ensemble simulation scripts
├── figures/                  # Publication figure generation
│
├── preprocess_figure_data.ipynb   # Compute metrics (run first)
└── generate_paper_figures.ipynb   # Generate visualizations (run second)
```

## Model Architectures

### U-Net
Baseline architecture adapted from Hu et al. (2025) using encoder-decoder structure with skip connections. Progressively downsamples vertical dimension while expanding feature space, with scalar outputs averaged and concatenated to vertically-resolved variables.

### Squeezeformer (1st Place)
Integrates convolutional and transformer components. Originally designed for automatic speech recognition, combines local context capture via depthwise convolutions with global dependency modeling through multi-head self-attention.

### Pure ResLSTM (2nd Place)
Multi-layer bidirectional LSTM with residual connections. Processes vertical profiles through 10 blocks of LSTM + layer normalization + GELU activation, embedding a physical prior of vertical locality.

### PAO Model (3rd Place)
Processes vertically-resolved and scalar variables separately before combining. Uses residual blocks with convolutional and transformer components, followed by bidirectional LSTM layers.

### ConvNeXt (4th Place)
Modern convolutional architecture competitive with vision transformers. Employs depthwise convolutions with large kernels, batch normalization, and residual connections across multiple stages.

### Encoder-Decoder LSTM (5th Place)
Uses encoder-decoder MLP to learn combined latent representation before recurrent processing. Bidirectional LSTM followed by GRU layer, breaking traditional vertical locality assumptions.

## Training Configurations

Each model can be trained with 5 different configurations:

1. **Standard** (`training_default`): Baseline using Kaggle-available input variables
2. **Confidence Loss** (`training_conf_loss`): Adds confidence head to predict loss magnitude (1st place team innovation)
3. **Difference Loss** (`training_diff_loss`): Adds loss term comparing vertical differences (2nd place team innovation)
4. **Multirepresentation** (`training_multirep`): Uses three parallel encodings of vertical profiles - level-wise normalization, column-wise normalization, and log-symmetric transformation (1st place team innovation)
5. **Expanded Inputs** (`training_v6`): Adds large-scale forcings, tendencies at previous timesteps (t-1, t-2), and latitude coordinates (following Hu et al. 2025)

## Getting Started

### Data

ClimSim dataset available at [HuggingFace](https://huggingface.co/LEAP). The paper uses the low-resolution dataset with real geography.

### Training

Each model's training directory contains:
- `conf/`: Hydra configuration files for different seeds
- `slurm/`: SLURM job submission scripts
- `train_{model}.py`: Training script
- `{model}.py`: Model architecture definition
- `wrap_model.py`: Wrapper for online inference (includes normalization)

**Note**: Code is provided for transparency and reproducibility, not as out-of-the-box software. You will need to adapt paths and configurations for your environment.

Example structure:
```bash
baseline_models/unet/training_default/
├── conf/
│   ├── config.yaml              # Base configuration
│   ├── config_seed_7.yaml       # Seed 7 variant
│   ├── config_seed_43.yaml      # Seed 43 variant
│   └── config_seed_1024.yaml    # Seed 1024 variant
├── slurm/
│   └── unet.sbatch              # Job submission script
├── train_unet.py                # Training script
├── unet.py                      # Model architecture
└── wrap_model.py                # Inference wrapper
```

### Online Testing

Online coupled simulations use FTorch for PyTorch-Fortran integration. See the [FTorch-based E3SM-MMF repository](https://github.com/zyhu-hu/E3SM_nvlab/tree/ftorch/climsim_scripts/perlmutter_scripts) for:
- E3SM-MMF setup with FTorch
- Model integration workflow
- Simulation configuration files

### Evaluation

Offline evaluation scripts in `evaluation/offline/` compute test set metrics. Online evaluation scripts in `evaluation/online/` analyze coupled simulation output.

For reproducing paper figures:
1. Run `preprocess_figure_data.ipynb` to compute expensive metrics and save results
2. Run `generate_paper_figures.ipynb` to generate all main and supplementary figures

## Requirements

- PyTorch (for training and inference)
- [NVIDIA Modulus](https://docs.nvidia.com/deeplearning/modulus/) (used during training)
- Hydra (configuration management)
- Standard scientific Python stack (numpy, xarray, matplotlib, etc.)

See individual model directories for specific dependencies.

## Citation

If you use this code or build upon this work, please cite the accompanying paper (citation to be added upon publication).

## References

- [ClimSim Kaggle Competition](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim)
- Hu et al. (2025). Stable Machine-Learning Parameterization of Subgrid Processes with Real Geography and Full-physics Emulation. [arXiv:2407.00124](https://arxiv.org/abs/2407.00124)
- [ClimSim Dataset on HuggingFace](https://huggingface.co/LEAP)
- [FTorch-based E3SM-MMF](https://github.com/zyhu-hu/E3SM_nvlab/tree/ftorch/climsim_scripts/perlmutter_scripts)

## License

See [LICENSE](LICENSE) file for details.
