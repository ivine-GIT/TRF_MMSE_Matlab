clearvars
close all
clc;

%%

% Parameters
fs = 64; % sampling rate
W = floor(2*fs); % windows length    % changed by ivine
eff_win_scale = 1;

Tmax = 375; % decoder lag (ms)
lag = ceil(Tmax*fs/1000); % decoder lag (samples)
plt_fig = 1;
count = 0;

count = count +1;
subject = 'sample_EEG.mat';

data_electrode = 12;
noise_electrode = 21;
attention = [1 1 1 1 2 2];

[s1, s2, eeg, noise, durations] = getData(subject, data_electrode, noise_electrode, Tmax);

pre_tr_order = [1 2];
tr_order = [3 6];
te_order = [4 5];

fprintf('MSE_init: [%i %i] Train: [%i %i] Test: [%i %i] ',pre_tr_order, tr_order, te_order);

dec_1_init = zeros(lag,1);
dec_2_init = zeros(lag,1);
MSE_1_init = eye(lag);
MSE_2_init = eye(lag);

for seg_id = pre_tr_order
  t = 1:durations(seg_id);
  eeg_train = eeg(t,seg_id);
  noise_train = noise(t,seg_id);
  K = floor(length(eeg_train)/(W*eff_win_scale));
  
  spkr_pr_tr_1 =  s1(t,seg_id);
  spkr_pr_tr_2 = s2(t,seg_id);
  
  start_idx = 1;
  for k = 1:K
    stop_idx = start_idx + W-1;
    e_d = eeg_train(start_idx:stop_idx);
    e_d = e_d - mean(e_d);
    
    e_n = noise_train(start_idx:stop_idx);
    e_n = e_n - mean(e_n);
    
    s_1 = spkr_pr_tr_1(start_idx:stop_idx);
    s_2 = spkr_pr_tr_2(start_idx:stop_idx);
    [dec_1_init, dec_2_init, MSE_1_init, MSE_2_init] = seq_LMMSE(e_d, e_n, s_1, s_2,...
                                                       dec_1_init, dec_2_init, MSE_1_init, MSE_2_init, lag);
    start_idx = start_idx + (W*eff_win_scale);
  end
end

sub_att_dec = [];
sub_unatt_dec = [];
m1_att_markers = [];
m2_att_markers = [];

att_dec = (dec_1_init + dec_2_init)/2;
unatt_dec = (dec_1_init + dec_2_init)/2;
MSE_a = (MSE_1_init + MSE_2_init)/2;
MSE_u = (MSE_1_init + MSE_2_init)/2;

for seg_id = tr_order
  t = 1:durations(seg_id);
  eeg_train = eeg(t,seg_id);
  noise_train = noise(t,seg_id);
  
  K = floor(length(eeg_train)/(W*eff_win_scale));
  
  if (1 == attention(seg_id))
    speaker_att =  s1(t,seg_id);
    speaker_unatt = s2(t,seg_id);
    
    start_idx = 1;
    for k = 1:K
      stop_idx = start_idx + W-1;
      e_d = eeg_train(start_idx:stop_idx);
      e_d = e_d - mean(e_d);
      
      e_n = noise_train(start_idx:stop_idx);
      e_n = e_n - mean(e_n);
      
      s_a = speaker_att(start_idx:stop_idx);
      s_u = speaker_unatt(start_idx:stop_idx);
      
      % Compute decoder with LMMSE
      [att_dec, unatt_dec, MSE_a, MSE_u] = seq_LMMSE(e_d, e_n, s_a, s_u, att_dec, unatt_dec, MSE_a, MSE_u, lag);
      
      sub_att_dec = [sub_att_dec att_dec];
      sub_unatt_dec = [sub_unatt_dec unatt_dec];
      
      % Compute Attention markers
      n1_a = getMarker(att_dec,1);
      p2_a = getMarker(att_dec,2);
      n1_u = getMarker(unatt_dec,1);
      p2_u = getMarker(unatt_dec,2);
      
      % Store markers
      tmp_m_a = abs(n1_a) + abs(p2_a);
      tmp_m_u = abs(n1_u) + abs(p2_u);
      
      % if m1 is the attended speaker, then the first column will have
      % attended marker and vice versa
      m1_att_markers = [m1_att_markers; [tmp_m_a tmp_m_u]];
      start_idx = start_idx + (W*eff_win_scale);
      clear tmp_m_a tmp_m_u
    end
    
  elseif (2 == attention(seg_id))
    speaker_att =  s2(t,seg_id);
    speaker_unatt = s1(t,seg_id);
    
    start_idx = 1;
    for k = 1:K
      stop_idx = start_idx + W-1;
      e_d = eeg_train(start_idx:stop_idx);
      e_d = e_d - mean(e_d);
      
      e_n = noise_train(start_idx:stop_idx);
      e_n = e_n - mean(e_n);
      
      s_a = speaker_att(start_idx:stop_idx);
      s_u = speaker_unatt(start_idx:stop_idx);
      
      % Compute decoder with LMMSE
      [att_dec, unatt_dec, MSE_a, MSE_u] = seq_LMMSE(e_d, e_n, s_a, s_u, att_dec, unatt_dec, MSE_a, MSE_u, lag);
      
      sub_att_dec = [sub_att_dec att_dec];
      sub_unatt_dec = [sub_unatt_dec unatt_dec];
      
      % Compute Attention markers
      n1_a = getMarker(att_dec,1);
      p2_a = getMarker(att_dec,2);
      n1_u = getMarker(unatt_dec,1);
      p2_u = getMarker(unatt_dec,2);
      
      tmp_m_a = abs(n1_a) + abs(p2_a);
      tmp_m_u = abs(n1_u) + abs(p2_u);
      
      % if m2 is the attended speaker, then the second column will have
      % attended marker and vice versa
      m2_att_markers = [m2_att_markers; [tmp_m_u tmp_m_a]];
      
      start_idx = start_idx + (W*eff_win_scale);
      clear tmp_m_a tmp_m_u
    end
  end
end

%% SVM
rng('default')

X = [m1_att_markers; m2_att_markers];
y = [-1*ones(size(m1_att_markers,1), 1); ones(size(m2_att_markers,1), 1)]; % m1 = -1;  m2 = +1
svm_Mdl = fitcsvm(X, y, 'KernelFunction','linear', 'Standardize',true);
ScoreSVMModel = fitPosterior(svm_Mdl, X, y);

% Compute the scores over a grid
d = 0.25; % Step size of the grid
[x1Grid, x2Grid] = meshgrid(min(X(:,1)): d: 2*max(X(:,1)), min(X(:,2)): d: 2*max(X(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];        % The grid
[~,scores1] = predict(svm_Mdl, xGrid); % The scores

%% Testing

dec_1 = (dec_1_init + dec_2_init)/2;
dec_2 = (dec_1_init + dec_2_init)/2;
MSE_1 = (MSE_1_init + MSE_2_init)/2;
MSE_2 = (MSE_1_init + MSE_2_init)/2;

test_markers = [];
true_label = [];

for seg_id = te_order
  t = 1:durations(seg_id);
  eeg_test = eeg(t,seg_id);
  noise_test = noise(t,seg_id);
  
  % Number of windows
  K = floor(length(eeg_test)/(W*eff_win_scale));
  
  % m1 = -1;  m2 = +1
  tmp_attn = 2*attention(seg_id)-3;
  true_label = [true_label; tmp_attn*ones(K,1)];
  
  spkr_te_1 =  s1(t,seg_id);
  spkr_te_2 = s2(t,seg_id);
  
  start_idx = 1;
  for k = 1:K
    if (1 == k)
      stop_idx = start_idx + W-1;
    else
      stop_idx = start_idx + (W*eff_win_scale)-1;
    end
    
    e_d = eeg_test(start_idx:stop_idx);
    e_n = noise_test(start_idx:stop_idx);
    s_1 = spkr_te_1(start_idx:stop_idx);
    s_2 = spkr_te_2(start_idx:stop_idx);
    
    % Compute decoder with LMMSE
    [dec_1, dec_2, MSE_1, MSE_2] = seq_LMMSE(e_d, e_n, s_1, s_2, dec_1, dec_2, MSE_1, MSE_2, lag);
    
    % Compute attention markers
    n1_dec1 = getMarker(dec_1,1);
    p2_dec1 = getMarker(dec_1,2);
    n1_dec2 = getMarker(dec_2,1);
    p2_dec2 = getMarker(dec_2,2);
    
    tmp_m1 = abs(n1_dec1) + abs(p2_dec1);
    tmp_m2 = abs(n1_dec2) + abs(p2_dec2);
    test_markers = [test_markers; [tmp_m1 tmp_m2]];
    
    start_idx = start_idx + (W*eff_win_scale);
    clear tmp_m1 tmp_m2
  end
end

%%
[y_test_pred, soft_op] = predict(ScoreSVMModel, test_markers);
% [y_test_pred, soft_op] = predict(svm_Mdl, test_markers);

if attention(1) == 1
  TP = sum(y_test_pred(y_test_pred==1) == true_label(y_test_pred==1));
  TN = sum(y_test_pred(y_test_pred==-1) == true_label(y_test_pred==-1));
  FP = sum(y_test_pred(y_test_pred==1) ~= true_label(y_test_pred==1));
  FN = sum(y_test_pred(y_test_pred==-1) ~= true_label(y_test_pred==-1));
elseif attention(1)== 2
  TP = sum(y_test_pred(y_test_pred==-1) == true_label(y_test_pred==-1));
  TN = sum(y_test_pred(y_test_pred==1) == true_label(y_test_pred==1));
  FP = sum(y_test_pred(y_test_pred==-1) ~= true_label(y_test_pred==-1));
  FN = sum(y_test_pred(y_test_pred==1) ~= true_label(y_test_pred==1));
end

precision = TP/(TP+FP);
recall = TP/(TP+FN);

misclassified = (y_test_pred ~= true_label);
acc = 100*sum(y_test_pred == true_label)/ length(true_label);
f1_scores = 100*2*(precision *recall)/(precision + recall);

tmp_acc_th = -1*(test_markers(:,1) - test_markers(:,2)).*true_label;
acc_th = 100*(sum(tmp_acc_th>0)/length(tmp_acc_th));
clear tmp_acc_th;

%%
X_test = test_markers;
tot_num_win = size(soft_op,1);
t = linspace(0, tot_num_win*W*eff_win_scale/fs, tot_num_win);

ft_size = 12;
ln_width = 1.2;
scale = 2;
color = [[ 0.114 0.44 1]; [1 0.41 0.153]];
%   color = ['r'; 'b'];

if(plt_fig)
  figure('pos',[600 50 600 650]);
  pos = [0.15 0.75 0.8 0.175]; % [left bottom width height]
  subplot('Position',pos);
  %     subplot(4,2,[1 2])
  plot(t, X_test(:,1), 'color', color(1,:), 'linewidth', ln_width)
  hold on
  
  plot(t, X_test(:,2), 'color', color(2,:), 'linewidth', ln_width)
  xlim([0 650])
  xlabel('time (s)', 'FontSize',ft_size,'Interpreter','latex')
  ylabel({'Attention'; 'marker'}, 'FontSize',ft_size, 'Interpreter','latex')
  
  [~,hObj] = legend({'spkr1','spkr2'}, 'FontSize',ft_size, 'location', 'northwest', 'Interpreter','latex');
  hL = findobj(hObj,'type','line');  % get the lines, not text
  set(hL,'linewidth', ln_width)
  set(gca, 'FontSize',ft_size, 'TickLabelInterpreter','latex')
  
  pos = [0.15 0.50 0.8 0.175]; % [left bottom width height]
  subplot('Position',pos);
  %     subplot(4,2,[3 4])
  plot(t, soft_op(:,1), 'color', color(1,:), 'linewidth', ln_width)
  ylim([-0.2 1.2])
  xlim([0 650])
  hold on
  plot(t, soft_op(:,2), 'color', color(2,:), 'linewidth', ln_width)
  
  tmp_label = true_label;
  tmp_label(tmp_label == -1) = 0;
  tmp_label = 1 - tmp_label;
  plot(t, tmp_label,'--', 'color', [0 0 0 ], 'linewidth', 2)
  
  xlabel('time (s)', 'FontSize',ft_size, 'Interpreter','latex')
  ylabel({'spkr1 att'; 'probability'}, 'FontSize',ft_size, 'Interpreter','latex')
  %     legend('spkr1','spkr2','spkr1: true attn','Interpreter','latex','location','Best')
  %     legend('boxoff');
  set(gca, 'FontSize',ft_size, 'TickLabelInterpreter','latex')
  
  m1_lim = max(max(abs(X(:,1))), max(abs(X_test(:,1))));
  m2_lim = max(max(abs(X(:,2))), max(abs(X_test(:,2))));
  
  pos = [0.15 0.1 0.35 0.28]; % [left bottom width height]
  subplot('Position',pos);
  %     subplot(4,2,[5 7])
  hold on
  h(1:2) = gscatter(X(:,1), X(:,2), y, color);
  
  h(3) = plot(X(svm_Mdl.IsSupportVector,1), X(svm_Mdl.IsSupportVector,2),'ko','MarkerSize', 8);    % Support vectors
  tmp_scores1 = reshape(scores1(:,2),size(x1Grid));
  contour(x1Grid, x2Grid, tmp_scores1, [0 0], 'k','linewidth', ln_width);    % Decision boundary
  axis([0 scale*m1_lim 0 scale*m2_lim]);
  
  title('Dec Boundary: Training','Interpreter','latex')
  xlabel('spkr1\_marker', 'FontSize',ft_size, 'Interpreter','latex')
  ylabel('spkr2\_marker', 'FontSize',ft_size, 'Interpreter','latex')
  %     legend({'spkr1','spkr2','Support Vectors','Decision Boundary'},'Location','Best','Interpreter','latex')
  legend({'spkr1','spkr2','Support Vectors'},'Location','southwest','Interpreter','latex', 'FontSize',ft_size )
  legend('boxoff');
  set(gca, 'FontSize',ft_size, 'TickLabelInterpreter','latex')
  
  pos = [0.6 0.1 0.35 0.28]; % [left bottom width height]
  subplot('Position',pos);
  %     subplot(4,2,[6 8])
  hold on
  h(1:2) = gscatter(X_test(:,1), X_test(:,2), true_label, color);
  scatter(X_test(misclassified, 1), X_test(misclassified, 2), 'k', 'o', 'linewidth', ln_width);
  contour(x1Grid, x2Grid, tmp_scores1, [0 0], 'k','linewidth',ln_width);    % Decision boundary
  axis([0 scale*m1_lim 0 scale*m2_lim]);
  
  title('Misclassification: Testing','Interpreter','latex')
  xlabel('spkr1\_marker', 'FontSize',ft_size, 'Interpreter','latex')
  ylabel('spkr2\_marker', 'FontSize',ft_size, 'Interpreter','latex')
  legend({'spkr1','spkr2','Misclassified'},'Location','southwest','Interpreter','latex', 'FontSize',ft_size )
  legend('boxoff');
  set(gca, 'FontSize',ft_size, 'TickLabelInterpreter','latex');
  
end

if(isnan(f1_scores))
  f1_scores = 0;
end

fprintf('f1score: %.2f , acc: %.2f , acc_th: %.2f\n', f1_scores, acc, acc_th);



