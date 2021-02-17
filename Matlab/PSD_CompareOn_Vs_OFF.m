%Daniel Sand



function PSD_CompareOn_Vs_OFF(params,scores2plot)

% need to open PPT first
toPPT('setPageFormat','16:9');

load(fullfile(params.subjInfoFolder,'Tscores.mat'));
removeRows=strcmp(Tscores.iVisit,'all')%remove all visits form the table
Tscores(removeRows,:)=[];

if params.analysis.RelventElec_Flag
  Tscores=analysis_subjElecs(Tscores,params);
end

%scores name
Scores=Tscores.Properties.VariableNames
for iScorelabel=params.num_metaDataNames: params.num_metaDataNames+params.num_freqNames+params.num_freqRatioNames-1 %reduce '_power_MeanPSD' from the title!
    Scores{1, iScorelabel}=Scores{1, iScorelabel}(1:end-14)
end


  sideType={'right','left'};
%     sideType=unique(Tscores.side);
for iSide=1:length(sideType)
    H_fig=figure
    rows =strcmp((Tscores.side),sideType(iSide))
    Tscores_side=Tscores(rows,:);
   
    LFP_Elec=params.chansLabels
            for iElec=1:1
 %    
         rows =':';
        Tscores_elec=Tscores_side(rows,:);
    
        subjList=params.analysis.subjNames2fit;
        for iSubjL=1:length(subjList)
       %      for iSubj=1:%need to be 6
            iSubj=subjList(iSubjL)
            rows =strcmp((Tscores_elec.reference),char(iSubj))
            Tscores_subj=Tscores_elec(rows,:);        
        
              GroupsLabel={'OFF','ON'};
              Graph_title=[' Subj: ',char(iSubj),' ;Side: ',char(sideType(iSide))]
              subplot(2,3,iSubjL)
              H_fig=boxPlot_groupsPlot(Tscores_subj,GroupsLabel,scores2plot,Graph_title,H_fig,params)
              
            end
    end
        set(gcf,'position',get(0,'screensize'))
    toPPT(H_fig,'pos','NH','Height%',230); 
     close all

end

end
