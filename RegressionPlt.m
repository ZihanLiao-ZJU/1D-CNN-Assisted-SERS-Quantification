function [rho,R2,BIAS,RMSEP,LOD] = RegressionPlt(Y_tra , Y_tra_pre, Y_tst , Y_tst_pre)
%% Initialization
% Preallocate arrays for calibration (index 1) and test (index 2) sets
rho  = zeros(2,1);   % Pearson correlation coefficient
R2   = zeros(2,1);   % Coefficient of determination
BIAS = zeros(2,1);   % Mean prediction bias
RMSEP = zeros(2,1);  % Root mean square error of prediction
LOD  = zeros(2,1);   % Limit of detection

%% Calibration set: performance metrics
% Compute R² and Pearson correlation coefficient for the calibration set
SS_res = sum((Y_tra - Y_tra_pre).^2);
SS_tot = sum((Y_tra - mean(Y_tra)).^2);
R2(1) = 1 - SS_res / SS_tot;
rho(1) = corr(Y_tra.', Y_tra_pre.');

% Slope of the calibration regression line (Y_tra → Y_tra_pre)
A = [ones(length(Y_tra),1),Y_tra.'] \ Y_tra_pre.';
A = A(2);

% ----------------- Optional: calibration scatter plot --------------------
% figure
% scatter(Y_tra, Y_tra_pre, 30, 'b', 'filled', 'DisplayName', 'Calibration Points');
% hold on
% plot([min(Y_tst), max(Y_tst)], [min(Y_tst), max(Y_tst)], ...
%     'k--', 'DisplayName', 'Ideal Line');
% hold on
%
% % Text annotations: correlation coefficient and R²
% text(min(Y_tra)+0.05*range(Y_tra), max(Y_tra_pre)-0.05*range(Y_tra_pre), ...
%      ['Corr. Coeff.: ', num2str(rho(1),'%4.3f')], 'FontSize', 15);
% text(min(Y_tra)+0.05*range(Y_tra), max(Y_tra_pre)-0.13*range(Y_tra_pre), ...
%      ['R^2: ', num2str(R2(1),'%4.3f')], 'FontSize', 15);
%
% xlabel('Actual Concentration (\mug/mL)');
% ylabel('Predicted Concentration (\mug/mL)');
% title('Calibration set','FontSize',12)
% xlim([min(Y_tra,[],"all"), max(Y_tra,[],"all")])
% ylim([min(Y_tra_pre,[],"all"), max(Y_tra_pre,[],"all")])
% legend('Location', 'southeast',"Box","off");
% set(gca, 'FontSize', 15);

% RMSEP, bias, and LOD for the calibration set
RMSEP(1) = sqrt(mean((Y_tra - Y_tra_pre).^2));
BIAS(1)  = mean((Y_tra_pre - Y_tra));

% LOD based on predicted responses of blank calibration samples
ind_blk   = (Y_tra == 0);         % Indices of blank samples
Y_pre_blk = Y_tra_pre(ind_blk);   % Predicted values for blanks
LOD(1)    = 3*std(Y_pre_blk)/A;

%% Test set: performance metrics
% Compute R² and Pearson correlation coefficient for the test set
SS_res = sum((Y_tst - Y_tst_pre).^2);
SS_tot = sum((Y_tst - mean(Y_tst)).^2);
R2(2) = 1 - SS_res / SS_tot;
rho(2) = corr(Y_tst.', Y_tst_pre.');

% Slope of the test-set regression line (Y_tst → Y_tst_pre)
A = [ones(length(Y_tst),1),Y_tst.'] \ Y_tst_pre.';
A = A(2);

% ----------------- Optional: test set regression with CI/PI --------------
% % Fit linear regression model: predicted vs. actual concentration
% mdl = fitlm(Y_tst, Y_tst_pre);  % Predicted values as dependent variable
% Y_line = linspace(min(Y_tst), max(Y_tst), 200)';
% [Y_fit, CI] = predict(mdl, Y_line);  % Confidence interval (default: 95%)
%
% % Prediction interval (wider than CI)
% [~, PI] = predict(mdl, Y_line, 'Prediction', 'observation');
%
% % Plot test-set regression with confidence and prediction bands
% figure; hold on;
%
% % Light red region: prediction interval (95% PI)
% fill([Y_line; flipud(Y_line)], [PI(:,1); flipud(PI(:,2))], ...
%     [1, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.4, 'DisplayName', '95% PI');
%
% % Darker red region: confidence interval (95% CI)
% fill([Y_line; flipud(Y_line)], [CI(:,1); flipud(CI(:,2))], ...
%     [1, 0.4, 0.4], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', '95% CI');
%
% % Regression line
% plot(Y_line, Y_fit, 'r-', 'LineWidth', 2, 'DisplayName', 'Fit Line');
%
% % Test data points
% scatter(Y_tst, Y_tst_pre, 30, 'k', 'filled', 'DisplayName', 'Test Points');
%
% % Ideal 1:1 line
% plot([min(Y_tst), max(Y_tst)], [min(Y_tst), max(Y_tst)], ...
%     'k--', 'DisplayName', 'Ideal Line');
%
% % Text annotations: correlation coefficient and R²
% text(min(Y_tst)+0.05*range(Y_tst), ...
%      max(max(PI,[],"all"),max(Y_tst_pre))-0.05*range(Y_tst_pre), ...
%     ['Corr. Coeff.: ', num2str(rho(2),'%4.3f')], 'FontSize', 15);
% text(min(Y_tst)+0.05*range(Y_tst), ...
%      max(max(PI,[],"all"),max(Y_tst_pre))-0.15*range(Y_tst_pre), ...
%     ['R^2: ', num2str(R2(2),'%4.3f')], 'FontSize', 15);
%
% % Axes and legend formatting
% xlabel('Actual Concentration (\mug/mL)');
% ylabel('Predicted Concentration (\mug/mL)');
% title('Test set','FontSize',12);
% xlim([min(Y_tst), max(Y_tst)]);
% ylim([min(min(PI,[],"all"),min(Y_tst_pre))*1.3, max(max(PI,[],"all"),max(Y_tst_pre))]);
% legend('Location', 'southeast',"Box","off", 'NumColumns', 1);
% set(gca, 'FontSize', 15);

% RMSEP, bias, and LOD for the test set
RMSEP(2) = sqrt(mean((Y_tst_pre - Y_tst).^2));
BIAS(2)  = mean((Y_tst_pre - Y_tst));

% LOD based on predicted responses of blank test samples
ind_blk   = (Y_tst == 0);          % Indices of blank samples
Y_pre_blk = Y_tst_pre(ind_blk);    % Predicted values for blanks
LOD(2)    = 3*std(Y_pre_blk)/A;
end
