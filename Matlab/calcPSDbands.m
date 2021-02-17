function PSDbandsRelPowerAll=calcPSDbands(PSD,freqs,funcParam,params,subj,side,elec)

switch funcParam
    case 'non-normalized'
        fFactor=ones(size(freqs));
    case 'normalized'
        fFactor=freqs;
end
PSD=PSD.*fFactor;


if params.pnina_normlizeFlag
    allBandsFreqIdxs=find(freqs>=params.Range2CalculateFrom(1) & freqs<=params.Range2CalculateFrom(2));
    PSD=PSD./sum(PSD(allBandsFreqIdxs));
    for iFreq=1:length(params.HighCutoffs)
        [lowFreqTh,hiFreqTh]=indivdualFreq(params,iFreq,subj,side,elec,params.indivdualFreqband);
        if lowFreqTh>=params.Range2CalculateFrom(1) & hiFreqTh<=params.Range2CalculateFrom(2)
            freqIdxs=find(freqs>=lowFreqTh & freqs<=hiFreqTh);
            PSDbandsRelPowerAll(iFreq)=mean(PSD(freqIdxs));
        end
    end
else
    
    for iFreq=1:length(params.HighCutoffs)
      [lowFreqTh,hiFreqTh]=indivdualFreq(params,iFreq,subj,side,elec,params.indivdualFreqband);
        if lowFreqTh>=params.Range2CalculateFrom(1) & hiFreqTh<=params.Range2CalculateFrom(2)
            freqIdxs=find(freqs>=lowFreqTh & freqs<=hiFreqTh);
            PSDbandsRelPower=sum(PSD(freqIdxs));
            allBandsFreqIdxs=find(freqs>=params.Range2CalculateFrom(1) & freqs<=params.Range2CalculateFrom(2));
        elseif  lowFreqTh>=params.Range2CalculateFrom(3) & hiFreqTh<=params.Range2CalculateFrom(4)
            freqIdxs=find(freqs>=lowFreqTh & freqs<=hiFreqTh);
            PSDbandsRelPower=sum(PSD(freqIdxs));
            allBandsFreqIdxs=all([((freqs>=params.Range2CalculateFrom(3) & freqs<=params.Range2CalculateFrom(4)) ),(freqs<=95 | freqs>=105)],2);
        else
            error('SHITT! - illegal ranges ya bitch')
            break
        end
        PSDbandsRelPowerAll(iFreq)=PSDbandsRelPower./sum(PSD(allBandsFreqIdxs));
    end
end
end