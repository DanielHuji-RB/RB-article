%Daniel Sand

function Tscores_new=analysis_subjElecs(Tscores,params)
goodRows=[];
for iFile=1:height(Tscores)
    
    iSubj=((Tscores.reference(iFile,:)));
    subjList=params.analysis.subjNames2fit;
    [a,iSubj_index]=ismember(iSubj,subjList);
    
        if strcmp(Tscores.side(iFile,:),'left')
            side=1;
        end
        if strcmp(Tscores.side(iFile,:),'right')
            side=2;
        end
        releventElec= params.analysis.subjElecs{1,iSubj_index}{side,1};
        if strcmp(Tscores.Elec(iFile,:),releventElec)
            goodRows=[goodRows iFile]
        end
        
end
Tscores_new=Tscores(goodRows,:);
end