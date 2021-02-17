function Table=findRatios(powerPerFreq)
tabIdx=0;
for i=1:length(powerPerFreq)
    for j=i+1:length(powerPerFreq)
        tabIdx=tabIdx+1;
        Table(tabIdx,1)=i;
        Table(tabIdx,2)=j;
        Table(tabIdx,3)=powerPerFreq(i)/powerPerFreq(j);
    end
end

end
