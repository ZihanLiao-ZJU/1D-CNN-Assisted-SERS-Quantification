%% 1D CNN Regression
clear

% Load spectral dataset
load data_Au_SERS.mat

% ------------------------ Data Preprocessing -----------------------------
% Convert training, validation, and test sets to dlarray format.
% Permute dimensions from (N × D × C) to (C × D × N) to match 'CBT' layout,
% where:
%   C = Channel dimension
%   B = Batch dimension
%   T = Time (or sequence) dimension
X_tra_tmp = dlarray(permute(X_tra,[3 2 1]), 'CBT'); 
Y_tra_tmp = dlarray(Y_tra.',"");

X_vad_tmp = dlarray(permute(X_vad,[3 2 1]), 'CBT');
Y_vad_tmp = dlarray(Y_vad.',"");

X_tst_tmp = dlarray(permute(X_tst,[3 2 1]), 'CBT');
Y_tst_tmp = dlarray(Y_tst.',"");

% ------------------------ Training Hyperparameters ------------------------
mb = 32;                                 % Mini-batch size
itersPerEpoch = ceil(N_tra/mb);          % Estimated iterations per epoch

% Optimization and training configuration
options = trainingOptions("adam", ...
    "MiniBatchSize", mb, ...
    "MaxEpochs", 500, ...
    "Shuffle","every-epoch", ...
    "ValidationData",{X_vad_tmp, Y_vad_tmp}, ...
    "ValidationFrequency", max(1, round(itersPerEpoch)), ...
    "L2Regularization", 1e-4, ...
    "InitialLearnRate", 1e-3, ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor", 0.5, ...
    "LearnRateDropPeriod", 500);

% ------------------------ Network Architecture ---------------------------
% 1D CNN designed for regression on Raman/SERS spectra.
layers = [ ...
    sequenceInputLayer(1, "MinLength", N_dim)

    % Convolution block 1
    convolution1dLayer(57, 8, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2,'Stride', 2, 'Padding','same')

    % Convolution block 2
    convolution1dLayer(37, 16, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2,'Stride', 2, 'Padding','same')

    % Convolution block 3
    convolution1dLayer(11, 32, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2,'Stride', 2, 'Padding','same')

    % Convolution block 4
    convolution1dLayer(3, 32, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2,'Stride', 2, 'Padding','same')

    % Feature extraction and regression head
    globalAveragePooling1dLayer
    fullyConnectedLayer(8, Name="feature")
    reluLayer
    fullyConnectedLayer(1, Name="output")];

% ------------------------ Model Loading or Training -----------------------
% Uncomment to train from scratch:
tic
net = trainnet(X_tra_tmp, Y_tra_tmp, layers, "mse", options);
toc

%% -------------------------- Regression Evaluation ------------------------
% Combine training and validation datasets for final regression fitting
X_tra_tmp = dlarray(permute([X_tra, X_vad], [3 2 1]), 'CBT');
Y_tra_tmp = [Y_tra, Y_vad];

% Predict target values for test and training+validation sets
Y_tst_pre = extractdata(predict(net, X_tst_tmp));
Y_tra_pre = extractdata(predict(net, X_tra_tmp));

% Compute regression metrics: ρ, R², bias, RMSEP, and LOD
[rho, R2, BIAS, RMSEP, LOD] = RegressionPlt( ...
    Y_tra_tmp, Y_tra_pre, Y_tst, Y_tst_pre);
