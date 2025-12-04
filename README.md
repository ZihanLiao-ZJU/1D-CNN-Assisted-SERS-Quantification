# 1D-CNN-Assisted SERS Quantification

This repository provides a complete MATLAB workflow for performing **quantitative regression** on Surface-Enhanced Raman Spectroscopy (SERS) data using a **1D Convolutional Neural Network (1D-CNN)**.

The workflow consists of:

1. Data Preparation  
2. Canonical Data Formatting  
3. dlarray Conversion  
4. CNN Architecture  
5. CNN Training  
6. Evaluation  
7. Full Pipeline Summary  

This single document explains the entire process in a unified Markdown format.

---

# 1. Data Preparation

The workflow begins with raw Raman/SERS spectral data and their corresponding target values (e.g., analyte concentrations).

## 1.1 Supported Input Formats

MATLAB-readable formats:

- `.mat`
- `.csv`, `.txt`
- `.xlsx`
- Any format readable by `load`, `readmatrix`, or `readtable`

The raw dataset must provide:

- `X_raw` — spectral matrix  
- `Y_raw` — target vector  

Supported layouts:

| Layout | Dimension | Description |
|--------|-----------|-------------|
| Spectra in rows | `N_samples × N_dim` | Each row is a spectrum |
| Spectra in columns | `N_dim × N_samples` | Each column is a spectrum |

The preparation script will correct orientation automatically.

---

## 1.2 Canonical Internal Format

After standardization, data are saved into `data_CNN.mat` using the following unified format:

### Spectra (`N_dim × N_samples`)

- `X_tra` — training spectra  
- `X_vad` — validation spectra  
- `X_tst` — test spectra  

### Targets (`1 × N_samples`)

- `Y_tra`  
- `Y_vad`  
- `Y_tst`  

### Metadata

- `N_dim`  
- `N_sam`  
- `N_tra`, `N_vad`, `N_tst`  
- optional: `len_wav` (wavenumber axis)

This file is the **only required input** for the CNN training script.

---

# 1.3 Data Preparation Pipeline

### Step 1 — Load raw data  
Use MATLAB functions such as:

```
load(...)
readmatrix(...)
readtable(...)
```

### Step 2 — Standardize orientation  
Convert into:

- Spectra: `[N_dim × N_samples]`  
- Targets: `[1 × N_samples]`  

### Step 3 — Randomly split dataset  
Common ratios:

- 60% training  
- 20% validation  
- 20% testing  

### Step 4 — Save canonical dataset  
Save:

```
X_tra, X_vad, X_tst
Y_tra, Y_vad, Y_tst
N_dim, N_sam
N_tra, N_vad, N_tst
```

into `data_CNN.mat`.

---

# 2. Data Formatting for CNN Training

## 2.1 Required dlarray Format

MATLAB `trainnet` requires 1D sequential input in **CBT** layout:

- **C** — Channel  
- **B** — Batch (samples)  
- **T** — Time / wavenumber axis  

Thus, the required input size is:

```
[1 × N_samples × N_dim]
```

---

## 2.2 Conversion Procedure

Given canonical spectra `[N_dim × N_samples]`:

1. Reshape to `[T × C × B]`  
2. Permute to `[C × B × T]`  
3. Wrap as dlarray:

```
dlarray(X_CBT, "CBT")
```

`drive_CNN.m` performs all conversion automatically.

---

# 3. CNN Architecture

The 1D-CNN structure consists of:

- Sequence input layer  
- Repeated blocks:  
  ```
  Conv → BatchNorm → ReLU → MaxPooling
  ```
- Global average pooling  
- Fully connected layer  
- Regression layer  

Kernel sizes and filter numbers are adjustable via parameters defined before constructing the network.

---

# 4. CNN Training Workflow (drive_CNN.m)

## 4.1 Loading the standardized dataset

The script automatically loads:

```
X_tra, X_vad, X_tst
Y_tra, Y_vad, Y_tst
N_dim, N_sam
N_tra, N_vad, N_tst
```

## 4.2 dlarray conversion

The script:

- Converts spectra into **CBT** dlarray format  
- Converts targets into column-vector dlarrays  
- Handles batching and shuffling  

No manual formatting is required.

---

## 4.3 Training Settings

Training uses MATLAB `trainnet` with:

- Loss: MSE  
- Regularization: L2  
- LR scheduling  
- Validation monitoring  

User-editable hyperparameters include:

- Batch size  
- Maximum epochs  
- Learning rate schedule  
- Regularization strength  

---

# 5. Evaluation

After training, predictions are made for:

- Calibration set  
- Test set  

Evaluation metrics include:

- Pearson correlation coefficient  
- R²  
- Bias  
- RMSEP  
- Limit of Detection (LOD), if implemented  

These metrics summarize the performance of the 1D-CNN on Raman/SERS quantification.

---

# 6. Full End-to-End Pipeline Summary

## Step 1 — Prepare Data
- Load raw spectra and targets  
- Standardize orientation  
- Split into training/validation/test  
- Save as `data_CNN.mat`  

## Step 2 — Format for CNN
- Convert to `[N_dim × N_samples]` format  
- Convert to `"CBT"` dlarray (done automatically)  

## Step 3 — Train CNN
- Run `drive_CNN.m`  
- CNN is constructed and trained automatically  

## Step 4 — Evaluate Model
- Predict concentrations  
- Compute regression metrics  
- Assess model performance  

## Step 5 — Reuse with Any Raman/SERS Dataset
Repeat:

1. Prepare raw data  
2. Save as `data_CNN.mat`  
3. Train using `drive_CNN.m`  
4. Evaluate results  

---
