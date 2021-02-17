%Daniel Sand

function genPSDdata(currData,subjInfo,params)


%% Local var
srate=subjInfo.srate;
%% define PSD constants
hwelch = spectrum.welch;
%% calc PSD per electrode data
PSDTH=params.PSD.TH*timeWindow/params.subSampRatio;

hwelch.SegmentLength=PSDTH;
currPSD = psd(hwelch,currData,'Fs',srate);

PSDdata.freqs = currPSD.Frequencies;
PSDdata.data{iElec,iTim} = currPSD.Data;

end
