# 1D-CNN-Assisted SERS Quantification

MATLAB code for quantitative regression of Surface-Enhanced Raman Spectroscopy (SERS) spectra with a one-dimensional convolutional neural network (1D-CNN). Given a set of spectra and their target values (e.g., analyte concentrations), the script trains a 1D-CNN regressor and reports calibration and test-set performance.

## Contents

- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Input data format](#input-data-format)
- [Network architecture](#network-architecture)
- [Training settings](#training-settings)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## Requirements

- MATLAB R2023b or later (the script uses `trainnet`)
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox (`corr`, `fitlm`)

## Repository structure

```
.
├── drive_CNN.m       % main script: data formatting, network definition, training, evaluation
├── RegressionPlt.m   % regression metrics and calibration / test-set plots
├── LICENSE
└── README.md
```

## Quick start

1. Prepare a MAT-file named `data_CNN.mat` in the format described below (no dataset is included in this repository).
2. Place it in the repository folder and run:

```matlab
>> drive_CNN
```

The script trains the network, prints the training time, and opens two figures (calibration set and test set). The metrics `rho`, `R2`, `BIAS`, `RMSEP` and `LOD` are left in the workspace.

## Input data format

`drive_CNN.m` loads `data_CNN.mat`, which must contain the following variables. Each column of `X_*` is one spectrum and each row is one wavenumber position.

| Variable | Size | Description |
|---|---|---|
| `X_tra` | `N_dim × N_tra` | training spectra |
| `X_vad` | `N_dim × N_vad` | validation spectra |
| `X_tst` | `N_dim × N_tst` | test spectra |
| `Y_tra` | `1 × N_tra` | training targets |
| `Y_vad` | `1 × N_vad` | validation targets |
| `Y_tst` | `1 × N_tst` | test targets |
| `N_dim` | scalar | number of wavenumber points |
| `N_sam` | scalar | total number of spectra |
| `N_tra`, `N_vad`, `N_tst` | scalar | number of training / validation / test spectra |

Splitting the raw data into training, validation and test sets is left to the user.

Inside the script, the spectra are reshaped and wrapped as `dlarray` objects with the `"CBT"` layout (channel × batch × spectral axis), as required by `trainnet` for 1D sequence input. If your spectra are stored in a different layout (spectra in rows, multi-channel, batch-first), the comment block *Templates for other data layouts* in `drive_CNN.m` shows the corresponding reshape/permute steps.

## Network architecture

```
sequenceInputLayer(1)
4 × [ convolution1dLayer → batchNormalizationLayer → reluLayer → maxPooling1dLayer(2, stride 2) ]
globalAveragePooling1dLayer
fullyConnectedLayer(featureDim) → reluLayer
fullyConnectedLayer(1)
```

The block parameters are set near the top of `drive_CNN.m`:

| Parameter | Default | Meaning |
|---|---|---|
| `kernelSizes` | `[57, 37, 11, 3]` | kernel size of each convolutional block |
| `numFilters` | `[8, 16, 32, 32]` | number of filters of each convolutional block |
| `featureDim` | `8` | size of the latent feature layer |

## Training settings

The network is trained with `trainnet` using the mean-squared-error loss and the Adam optimizer:

| Option | Value |
|---|---|
| Mini-batch size | 32 |
| Max epochs | 500 |
| Shuffle | every epoch |
| Initial learning rate | 1e-3 |
| Learning-rate schedule | piecewise, drop factor 0.5 every 500 epochs |
| L2 regularization | 1e-4 |
| Validation | `X_vad`, `Y_vad`, evaluated once per epoch |

## Evaluation

After training, the training and validation sets are combined into a calibration set, and predictions are made for the calibration and test sets. `RegressionPlt.m` computes, for each set (index 1 = calibration, 2 = test):

| Output | Definition |
|---|---|
| `rho` | Pearson correlation coefficient between actual and predicted values |
| `R2` | coefficient of determination |
| `BIAS` | mean of (predicted − actual) |
| `RMSEP` | root-mean-square error of prediction |
| `LOD` | 3 × standard deviation of the predictions for blank samples (target = 0) |

It also plots the calibration scatter and the test-set regression line with 95% confidence and prediction intervals. Axis labels assume concentrations in µg/mL.

## Citation

If you use this code in your research, please cite:

> Guo, L., Liao, Z., & Yang, T. (2026). Tailored Deep Learning–Assisted In Situ SERS: Overcoming Surface Irregularities-Induced Large Signal Variation on Biological Tissues. *Analytical Chemistry*.

```bibtex
@article{guo2026tailored,
  title     = {Tailored Deep Learning--Assisted In Situ SERS: Overcoming Surface Irregularities-Induced Large Signal Variation on Biological Tissues},
  author    = {Guo, Ling and Liao, Zihan and Yang, Tianxi},
  journal   = {Analytical Chemistry},
  year      = {2026},
  publisher = {ACS Publications}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
