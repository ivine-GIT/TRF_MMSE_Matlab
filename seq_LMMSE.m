function [att, unatt, MSE_new_a, MSE_new_u] =  seq_LMMSE(e_d, e_n, s_a, s_u, att_prev, unatt_prev, MSE_a, MSE_u, lag)

% Lagged speech matrices
S_a = zeros(length(s_a),lag);
S_u = zeros(length(s_a),lag);

for m = 0:lag-1
  if m > 0
    S_a(:,m+1) = [zeros(m,1) ; s_a(1:end-m,1)];
    S_u(:,m+1) = [zeros(m,1) ; s_u(1:end-m,1)];
  elseif m == 0
    S_a(:,m+1) = s_a;
    S_u(:,m+1) = s_u;
  end
end

if 1
  S_a = S_a(lag+1:end,:);
  S_u = S_u(lag+1:end,:);
  
  e_d = e_d(lag+1:end);
  e_n = e_n(lag+1:end);
end

%Noise correlation
C_WW = var(e_n)*eye(length(e_n));

K_gain_a = MSE_a*S_a.'*(inv(C_WW + S_a*MSE_a*S_a.'));
att = att_prev + K_gain_a*(e_d - S_a*att_prev);
MSE_new_a = (eye(lag) - K_gain_a*S_a)*MSE_a;

K_gain_u = (MSE_u*S_u.')*(inv(C_WW + S_u*MSE_u*S_u.'));
unatt = unatt_prev + K_gain_u*(e_d - S_u*unatt_prev);
MSE_new_u = (eye(lag) - K_gain_u*S_u)*MSE_u;
end