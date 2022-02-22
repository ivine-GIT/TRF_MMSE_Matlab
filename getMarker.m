function Final_Peak = getMarker(x,mode)

% mode=1 checks N100 point
% mode=2 checks P200 point

% Increase the sampling rate for high precision
U = 5; %upsampling rate
fs = 64*U;
x_up = interp(x,U);

if mode == 1
  lower_bound = floor(75*fs/1000);
  upper_bound = floor(135*fs/1000);
  
  [val,ind] = max(abs(x_up(lower_bound:upper_bound)));
  
  if x_up(ind+lower_bound-1)<0
    n1_peak = val;
  else
    n1_peak = 0;
  end
  
  Final_Peak = n1_peak;
  
elseif mode == 2
  lower_bound = floor(175*fs/1000);
  upper_bound = floor(250*fs/1000);
  
  [val,ind] = max(abs(x_up(lower_bound:upper_bound)));
  
  if x_up(ind+lower_bound-1)>0
    p2_peak = val;
  else
    p2_peak = 0;
  end
  
  Final_Peak = p2_peak;
end
end