%Daniel Sand

function compValuesNames=getScoresNames_activa(params)

PSDbandsPowerRatioTable=findRatios(zeros(size(params.freqNames)));
L_freq=length(params.freqNames);
L_ratio=length(PSDbandsPowerRatioTable);

% Generate freqNames
freqNames=params.freqNames;
for iName=L_freq+1:L_freq+L_ratio
    freqNames{iName}=[freqNames{PSDbandsPowerRatioTable(iName-L_freq,1)},'_',freqNames{PSDbandsPowerRatioTable(iName-L_freq,2)},'_Ratio'];
end
for iName=1:L_freq+L_ratio
compValuesNames{iName,1}=[freqNames{iName},'_power_MeanPSD'];
end
