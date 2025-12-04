%% ========================================================================
%  1D CNN Regression for Raman / SERS Spectra
%  - Single-target regression using a 1D convolutional neural network
%  - Inputs are converted to dlarray with layout 'CBT':
%       C : channel dimension
%       B : batch (sample) dimension
%       T : time / spectral dimension
% ========================================================================

clear; clc;

%% ====================== Load Spectral Dataset ============================
% The MAT-file is expected to contain:
%   N_dim  : number of spectral variables (wavenumber points)
%   N_sam  : total number of spectra
%   N_tra  : number of training samples
%   N_vad  : number of validation samples
%   N_tst  : number of test samples
%
%   X_tra  : training spectra     [N_dim × N_tra]
%   X_vad  : validation spectra   [N_dim × N_vad]
%   X_tst  : test spectra         [N_dim × N_tst]
%
%   Y_tra  : training targets     [1 × N_tra]
%   Y_vad  : validation targets   [1 × N_vad]
%   Y_tst  : test targets         [1 × N_tst]
%
% Each column of X_* corresponds to one spectrum; rows correspond to
% wavenumber positions.

load data_CNN.mat

% If needed, the following can be used to infer sizes:
% [N_dim, N_tra] = size(X_tra);
% [~,     N_vad] = size(X_vad);
% [~,     N_tst] = size(X_tst);

%% ================== Data Formatting and dlarray Layout ===================
% Goal: convert spectral matrices to dlarray with layout 'CBT', where the
% final size is:
%   [C × B × T] = [numChannels × numSamples × sequenceLength]
%
% For the current Raman / SERS dataset:
%   - Spectra are stored as [N_dim × N_samples] (columns = samples).
%   - There is a single spectral channel.
%
% Conversion steps for this layout:
%   1) Reshape [N_dim × N_samples] to [T × C × B]
%   2) Permute to [C × B × T]
%   3) Wrap as dlarray with labels 'CBT'

% ---------------------- Training set -------------------------------------
X_tra_3d  = reshape(X_tra, [N_dim, 1, N_tra]);          % [T × C × B]
X_tra_tmp = dlarray(permute(X_tra_3d, [2 3 1]), 'CBT'); % [C × B × T]

% ---------------------- Validation set -----------------------------------
X_vad_3d  = reshape(X_vad, [N_dim, 1, N_vad]);          % [T × C × B]
X_vad_tmp = dlarray(permute(X_vad_3d, [2 3 1]), 'CBT'); % [C × B × T]

% ---------------------- Test set -----------------------------------------
X_tst_3d  = reshape(X_tst, [N_dim, 1, N_tst]);          % [T × C × B]
X_tst_tmp = dlarray(permute(X_tst_3d, [2 3 1]), 'CBT'); % [C × B × T]

% Targets: convert to column vectors and then to dlarray
Y_tra_tmp = dlarray(Y_tra.', "");   % [N_tra × 1]
Y_vad_tmp = dlarray(Y_vad.', "");   % [N_vad × 1]
Y_tst_tmp = dlarray(Y_tst.', "");   % [N_tst × 1]

% ---------------------- Templates for other data layouts -----------------
% If spectra are stored in a different layout, the conversion to 'CBT'
% should be modified accordingly. Common cases are:
%
% Case A: rows are spectra, X = [N_samples × N_dim]
%   X_row      : [B × T]
%   X_row_3d   = reshape(X_row, [B, 1, T]);              % [B × C × T]
%   X_dlarray  = dlarray(permute(X_row_3d, [2 1 3]), ...
%                        'CBT');                         % [C × B × T]
%
% Case B: multi-channel spectra, X = [N_dim × N_samples × numChannels]
%   X_mc       : [T × B × C]
%   X_mc_perm  = permute(X_mc, [3 2 1]);                 % [C × B × T]
%   X_dlarray  = dlarray(X_mc_perm, 'CBT');
%
% Case C: batch-first, multi-channel, X = [N_samples × N_dim × numChannels]
%   X_bf       : [B × T × C]
%   X_bf_perm  = permute(X_bf, [3 1 2]);                 % [C × B × T]
%   X_dlarray  = dlarray(X_bf_perm, 'CBT');
%
% In all cases, the final dlarray for trainnet must be of size
%   [C × B × T]
% with labels 'CBT'.

%% ==================== Training Hyperparameters ==========================
mb = 32;                              % mini-batch size
itersPerEpoch = ceil(N_tra / mb);     % iterations per epoch (for validation)

options = trainingOptions("adam", ...
    "MiniBatchSize",        mb, ...
    "MaxEpochs",            500, ...
    "Shuffle",              "every-epoch", ...
    "ValidationData",       {X_vad_tmp, Y_vad_tmp}, ...
    "ValidationFrequency",  max(1, round(itersPerEpoch)), ...
    "L2Regularization",     1e-4, ...
    "InitialLearnRate",     1e-3, ...
    "LearnRateSchedule",    "piecewise", ...
    "LearnRateDropFactor",  0.5, ...
    "LearnRateDropPeriod",  500);

%% ============= Network Architecture Hyperparameters (configurable) ======
% Convolution kernel sizes (one per block, in order from shallow to deep)
kernelSizes = [57, 37, 11, 3];

% Number of filters in each convolutional layer
numFilters  = [8, 16, 32, 32];

% Dimension of the latent feature representation
featureDim  = 8;

% (Optional) consistency check
assert(numel(kernelSizes) == numel(numFilters), ...
    "kernelSizes and numFilters must have the same length.");

%% ====================== Network Architecture ============================
% 1D CNN for regression on Raman / SERS spectra.
% The sequenceInputLayer uses a single input channel; the minimum sequence
% length is set according to N_dim.

layers = [ ...
    sequenceInputLayer(1, "MinLength", N_dim)

    % Convolution block 1
    convolution1dLayer(kernelSizes(1), numFilters(1), Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, "Stride", 2, "Padding", "same")

    % Convolution block 2
    convolution1dLayer(kernelSizes(2), numFilters(2), Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, "Stride", 2, "Padding", "same")

    % Convolution block 3
    convolution1dLayer(kernelSizes(3), numFilters(3), Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, "Stride", 2, "Padding", "same")

    % Convolution block 4
    convolution1dLayer(kernelSizes(4), numFilters(4), Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, "Stride", 2, "Padding", "same")

    % Feature extraction and regression head
    globalAveragePooling1dLayer
    fullyConnectedLayer(featureDim, Name="feature")
    reluLayer
    fullyConnectedLayer(1, Name="output")];

%% ======================= Model Training =================================
tic
net = trainnet(X_tra_tmp, Y_tra_tmp, layers, "mse", options);
toc

%% ===================== Regression Evaluation ============================
% Combine training and validation sets as a single calibration set
X_tra_all = [X_tra, X_vad];              % [N_dim × (N_tra + N_vad)]
Y_tra_all = [Y_tra, Y_vad];              % [1 × (N_tra + N_vad)]
N_tra_all = size(X_tra_all, 2);

X_tra_all_3d  = reshape(X_tra_all, [N_dim, 1, N_tra_all]);       % [T × C × B]
X_tra_all_tmp = dlarray(permute(X_tra_all_3d, [2 3 1]), "CBT");  % [C × B × T]

% Predictions for test set and combined calibration set
Y_tst_pre = extractdata(predict(net, X_tst_tmp));       % test predictions
Y_tra_pre = extractdata(predict(net, X_tra_all_tmp));   % calibration predictions

% Regression metrics (user-defined function)
[rho, R2, BIAS, RMSEP, LOD] = RegressionPlt( ...
    Y_tra_all, Y_tra_pre, Y_tst, Y_tst_pre);
