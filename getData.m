function [s1, s2, data_eeg, noise_eeg, durations] = getData_AEP(subject, data_electrode, noise_electrode, Tmax)

fs = 64; % sampling rate
fs_if = 1000; % sampling rate of AEP

%% Filter: Bandpass

% Larger BW used to take care of the transition period.
% Essentially, transition starts around 8 Hz

N = 7;
Rs = 50;
fstop1 = 1;
fstop2 = 10;

Wn = [fstop1 fstop2]/(fs/2);

[z,p,k] = cheby2(N,Rs,Wn,'bandpass');
[fil_bp_nr,fil_bp_dr] = zp2tf(z,p,k);

% % % dirac = [1 zeros(1,255)];
% % % h = filter(fil_bp_nr,fil_bp_dr,dirac);
% % % figure;
% % % freqz(h);
% % % % figure;
% % % % zplane(fil_bp_nr,fil_bp_dr);

clear N Rs fstop1 fstop2 Wn z p k ;

%% Filter: Bandpass

% Larger BW used to take care of the transition period.
% Essentially, transition starts around 8 Hz

N = 7;
Rs = 50;
fstop1 = 1;
fstop2 = 10;
Wn = [fstop1 fstop2]/(fs/2);

[z,p,k] = cheby2(N,Rs,Wn,'bandpass');
[fil_1Hz_9Hz_nr, fil_1Hz_9Hz_dr] = zp2tf(z,p,k);

% % dirac = [1 zeros(1,255)];
% % h = filter(fil_1Hz_9Hz_nr,fil_1Hz_9Hz_dr,dirac);
% % figure;
% % freqz(h);
% % % % figure;
% % % % zplane(fil_1Hz_9Hz_nr,fil_1Hz_9Hz_dr);

clear N Rs fstop1 fstop2 Wn z p k ;

%% Filter: Low-Pass
% % % 
% % % N = 10;
% % % Rs = 60;
% % % fstop = 9;
% % % Wn = fstop/(fs/2);
% % % 
% % % [z,p,k] = cheby2(N,Rs,Wn,'low');
% % % [fil_lp_9Hz_nr,fil_lp_9Hz_dr] = zp2tf(z,p,k);
% % % 
% % % % dirac = [1 zeros(1,255)];
% % % % h = filter(fil_lp_9Hz_nr,fil_lp_9Hz_dr,dirac);
% % % % figure;
% % % % freqz(h);
% % % % % % figure
% % % % % % zplane(fil_lp_9Hz_nr,fil_lp_9Hz_dr);


clear N Rs fstop Wn z p k ;

%% Loading Speech Envelope
SpeechPath = './speech_env/';
SpeechStruct = dir(SpeechPath);
SpeechIndex = [SpeechStruct.isdir];
SpeechFiles = {SpeechStruct(~SpeechIndex).name};

% Obtain durations of speech segments
durations = zeros(1,length(SpeechFiles));
index=1;
for s=SpeechFiles
  [y,~] = audioread(char(strcat(SpeechPath,s)));
  durations(index)= length(y);
  index = index+1;
end

% Init s1 and s2
s1 = zeros(max(durations),length(SpeechFiles));
s2 = zeros(max(durations),length(SpeechFiles));

% Load speech
dim=1;
for s=SpeechFiles
  [y,~] = audioread(char(strcat(SpeechPath,s)));
  s1(1:durations(dim),dim) = filtfilt(fil_1Hz_9Hz_nr, fil_1Hz_9Hz_dr, y(:,1));
  s2(1:durations(dim),dim) = filtfilt(fil_1Hz_9Hz_nr, fil_1Hz_9Hz_dr, y(:,2));
  dim = dim +1;
end

%% Loading Neural Response

% Load random subject data
EEGdata = load(subject);
Segs = fieldnames(EEGdata);
eegs = [];

% Concatenate all segments in EEGdata
noise_eeg = zeros(max(durations),length(Segs));
data_eeg = zeros(max(durations),length(Segs));

dim = 1;
for s=Segs.'
  EEG = getfield(EEGdata,char(s),'eeg_env');
  
  EEG_data = EEG(1:durations(dim),data_electrode);
  EEG_data = filtfilt(fil_bp_nr, fil_bp_dr, EEG_data);
  data_eeg(1:length(EEG_data),dim) = EEG_data;
  
  EEG_noise = EEG(1:durations(dim),noise_electrode);
  EEG_noise = filtfilt(fil_bp_nr, fil_bp_dr, EEG_noise);
  noise_eeg(1:length(EEG_noise),dim) = EEG_noise;
  
  dim = dim +1 ;
end
end