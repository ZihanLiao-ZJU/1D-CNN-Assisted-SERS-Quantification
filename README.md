# 1D-CNN-Assisted-SERS-Quantification
This repository provides a complete workflow for performing quantitative regression on Raman or Surface-Enhanced Raman Spectroscopy (SERS) data using a 1D Convolutional Neural Network (1D-CNN). The workflow consists of:

Data preparation
Load raw Raman/SERS spectra from any MATLAB-readable file, standardize formatting, split into training/validation/test sets, and save a unified .mat file.

CNN training (drive_CNN.m)
Load the prepared .mat file, convert the inputs into dlarray format, and train the 1D-CNN.

This README describes all required steps in detail.

1. Data Preparation
1.1 Acceptable Raw Data Formats

Raw spectral data may be loaded from any MATLAB-readable file, including:

.mat

.csv, .txt

.xlsx

Any format that can be read by readmatrix, readtable, or load

The raw data must include:

A spectral matrix X_raw

A corresponding target vector Y_raw (e.g., analyte concentrations)

Two common input layouts are supported:

Rows = spectra, columns = wavenumber points
X_raw has size [N_samples × N_dim]

Columns = spectra, rows = wavenumber points
X_raw has size [N_dim × N_samples]

The data preparation script will automatically standardize the orientation.

1.2 Standard Internal Format

To ensure compatibility with the CNN training script, all data are converted into a single canonical format and stored inside a .mat file.

All spectra are stored column-wise as:

X_tra — training spectra, [N_dim × N_tra]

X_vad — validation spectra, [N_dim × N_vad]

X_tst — test spectra, [N_dim × N_tst]

Target values are stored row-wise as:

Y_tra — [1 × N_tra]

Y_vad — [1 × N_vad]

Y_tst — [1 × N_tst]

Metadata included in the .mat file:

N_dim — number of wavenumber points

N_sam — total number of samples

N_tra, N_vad, N_tst — sample counts per subset

Optional:

len_wav — wavenumber axis

1.3 Data Preparation Pipeline

The recommended workflow is:

Load raw data
Example sources include MAT, CSV, TXT, Excel, or instrument output files.

Standardize orientation
Spectra are converted into the format [N_dim × N_samples].
Targets are converted into [1 × N_samples].

Randomly split dataset
Typical split ratios:

Training: ~70%

Validation: ~15%

Test: ~15%

Save standard dataset
Save all variables into a single file data_CNN.mat:

X_tra, X_vad, X_tst
Y_tra, Y_vad, Y_tst
N_dim, N_sam
N_tra, N_vad, N_tst


This .mat file will be the only required input for drive_CNN.m.

2. Data Formatting for CNN Training
2.1 dlarray Format Requirements

MATLAB’s trainnet requires sequential data to be passed in the dlarray format using the "CBT" layout:

C — Channel dimension

B — Batch (sample) dimension

T — Time/sequence dimension (wavenumber axis)

For Raman/SERS spectra, the final input must have shape:

[1×N_samples×N_dim]
[1×N_samples×N_dim​]

2.2 Creating "CBT" dlarray Inputs

Given a spectral matrix in the canonical format [N_dim × N_samples], the conversion is:

Reshape to 3D [T × C × B]

Permute to [C × B × T]

Wrap with dlarray(..., "CBT")

This conversion is performed inside the training script automatically.

3. CNN Training Workflow (drive_CNN.m)

The CNN training script requires only one input:

data_CNN.mat


which is generated in the data preparation step.

3.1 Loading Data

drive_CNN.m begins by loading:

X_tra, X_vad, X_tst
Y_tra, Y_vad, Y_tst
N_dim, N_sam
N_tra, N_vad, N_tst

3.2 Converting Data to dlarray

All spectral matrices are converted to "CBT" dlarray format.
Targets are converted into column-vector dlarrays.

No manual intervention is needed as long as the dataset is saved in the prescribed format.

4. CNN Architecture

The model is a 1D convolutional neural network that includes:

A sequence input layer

A series of convolution → batch normalization → ReLU → max pooling blocks

A global average pooling layer

A fully connected feature layer

A final regression layer

Convolution kernel sizes and number of filters are user-configurable through parameters defined before network construction.

5. Training and Validation

Training is performed using trainnet with the mean squared error loss. The script handles:

Mini-batching

Training/validation shuffling

Learning-rate scheduling

L2 regularization

Validation monitoring

Hyperparameters such as batch size, maximum epochs, and learning rate schedule can be freely adjusted.

6. Evaluation

After training:

The training and validation sets may be merged to form a calibration set.

Predictions are computed for:

Calibration data

Test data

A user-defined evaluation function (RegressionPlt.m) computes:

Pearson correlation coefficient

Coefficient of determination (R²)

Bias

RMSEP

Limit of Detection (LOD), if implemented

These results quantify the regression performance of the 1D-CNN.

7. Summary of the Full Pipeline

Prepare data

Load raw Raman/SERS spectra (X_raw) and target values (Y_raw)

Standardize formatting

Split into training/validation/testing sets

Save everything into data_CNN.mat

Train CNN

Run drive_CNN.m

Data are automatically converted into dlarray format

1D-CNN is trained using the specified network parameters and hyperparameters

Evaluate model

Generate predictions on test and calibration datasets

Compute regression statistics

Analyze CNN performance for Raman/SERS quantification tasks

This workflow allows seamless reuse with any Raman/SERS dataset that MATLAB can load, ensuring consistent preprocessing and CNN training procedures.
