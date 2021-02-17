%Daniel Sand

function [meanVisits,stdVisits]=freqStatsVisits(maxFreqValues_Right,iSubjL)
for iOnOff=1:2
    A=(squeeze(maxFreqValues_Right(:,iOnOff,iSubjL,:)));
    A(A == 0) = NaN;
    meanVisits(iOnOff,:)=nanmean(A);
    stdVisits(iOnOff,:)=nanstd(A);
    
end

end