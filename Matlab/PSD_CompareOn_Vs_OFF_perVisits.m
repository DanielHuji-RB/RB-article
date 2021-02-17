
%Daniel Sand


function PSD_CompareOn_Vs_OFF_perVisits(params,scores2plot)

reducing_ElectrodePower=0;

% need to open PPT first
toPPT('setPageFormat','16:9');

load(fullfile(params.subjInfoFolder,'Tscores.mat'));
Scores=Tscores.Properties.VariableNames

for iScorelabel=params.num_metaDataNames: params.num_metaDataNames+params.num_freqNames+params.num_freqRatioNames-1 %reduce '_power_MeanPSD' from the title!
    Scores{1, iScorelabel}=Scores{1, iScorelabel}(1:end-14)
end



sideType={'right','left'};
%     sideType=unique(Tscores.side);
for iScore=1:length(scores2plot)
    for iSide=1:length(sideType)
        H_fig=figure
        rows =strcmp((Tscores.side),sideType(iSide))
        Tscores_side=Tscores(rows,:);
        if reducing_ElectrodePower
       Tscores_side= reducing_ElectrodePower(Tscores_side);
        end      
        
        LFP_Elec=params.chansLabels
        for iElec=1:length(LFP_Elec)
            rows =strcmp((Tscores_side.Elec),LFP_Elec(iElec))
            Tscores_elec=Tscores_side(rows,:);
            
            for iSubj=1:1%need to be 6
                Tscores_subj=Tscores_elec(:,:);
                
                GroupsLabel={'OFF','ON'};
                Graph_title=[ Scores{1, scores2plot(iScore)} ' per visits; Elec: ',char(LFP_Elec(iElec)),' ;Side: ',char(sideType(iSide))]
                subplot(2,3,iElec)
                H_fig=PSDPerVisit_plot(Tscores_subj,params,GroupsLabel, scores2plot(iScore),Graph_title,H_fig,params.analysis.subjNames2fit)
                
            end
        end
        set(gcf,'position',get(0,'screensize'))
        toPPT(H_fig,'pos','NH','Height%',230); % NEH = NorthEastHalf%  'help getPosParameters'
        close all
        
    end
    
end
end
