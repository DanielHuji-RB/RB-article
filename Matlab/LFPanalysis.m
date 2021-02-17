
%Daniel Sand
function [h_spectogram,h_psd,T2]=LFPanalysis(currData,params,side,subj,elec,visit,onOff,iPart)
srate=params.srate;
hwelch = spectrum.welch;

hwelch.segmentLength = params.PSD.timeWindow*srate;


%% plot psd
currPSD=psd(hwelch,currData,'Fs',srate);

%% normalizion
if strcmp(params.freqNormlizedFlag,'non-normalized')
    fFactor=ones(size(currPSD.Frequencies));
else if strcmp(params.freqNormlizedFlag,'normalized')
        fFactor=currPSD.Frequencies;
    end
end
currPSD_Data=(currPSD.Data).*fFactor;
if params.notchBand_nan_flag==1
    idx_notch = find((45<currPSD.Frequencies)&(currPSD.Frequencies < 58))
    currPSD_Data(idx_notch)=nan
      idx_notch = find((99<currPSD.Frequencies)&(currPSD.Frequencies < 110))
    currPSD_Data(idx_notch)=nan
     idx_notch = find((149<currPSD.Frequencies)&(currPSD.Frequencies < 155))
    currPSD_Data(idx_notch)=nan
           
end


bandValue=PSDinAdaptivePeak(currPSD.Data,currPSD.Frequencies,params.freqNormlizedFlag,params);%this function is not update to use max freq

PSDbandsRelPower(:,1)=calcPSDbands(currPSD.Data,currPSD.Frequencies,params.freqNormlizedFlag,params,subj,side,elec); %
PSDbandsPowerRatioTable=findRatios(PSDbandsRelPower);
iTim=1;
compValuesOverTime(:,iTim)=[PSDbandsRelPower;PSDbandsPowerRatioTable(:,3)];

%% plot spectogram
h_spectogram=figure; set(h_spectogram,'Visible', 'off'); %stops the current figure from popping up.

window=srate*params.PSD.timeWindow;
noverlap=srate/2; 
f=0.5:80;
spectrogram(currData,window,noverlap,f,srate,'yaxis');
if params.zScoreFlag==1
    title('zScore')
end

%% building Tscore table
compValuesNames=getScoresNames_activa(params);
%%
%

T = table(compValuesOverTime,'RowNames',compValuesNames);
TA = table2array(T);%transpose table
T2 = array2table(TA.');%transpose table
T2.Properties.VariableNames = T.Properties.RowNames;

%plot PSD
h_psd=figure; set(h_psd,'Visible', 'off'); %stops the current figure from popping up.
if params.notchBand_nan_flag==1
    idx_notch = find((45<currPSD.Frequencies)&(currPSD.Frequencies < 58))
    currPSD_Data(idx_notch)=nan
%     [F,TF]=fillmissing( currPSD_Data(idx_notch),'spline','SamplePoints')
           
end
plot(currPSD.Frequencies,log(currPSD_Data))
% xlim([0 80]);


%% EpisodeBursts calculation
T2= episodeBursts(T2,currData,params,subj,side,elec,visit,onOff,iPart);


end