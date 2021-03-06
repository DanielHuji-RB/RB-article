%Daniel Sand


function T2=episodeBursts(T2,rawData,params,subj,side,elec,visit,OnOff,iPart)
plotFigs_flag=0;
plotDistrubtionFiguresFlag=0;
saveHilbert=0;
pathHilbert='C:\Us\';


if params.flag.splitData_Flag== 1 
    rawData_fig=figure;set(rawData_fig, 'Visible', 'off');
    plot(rawData)
    tempZ=zeros(params.reduceStartSession,1);
    rawData=[tempZ;rawData];
end
numberOfscores_beforeEpisodes=width(T2);
percentiles_paramter=65;% 
toPPT('setPageFormat','16:9');
h_distrubtion= figure; set(h_distrubtion, 'Visible', 'off');
episodeBursts_S=struct;

s=struct;%

for iFreqBand=1:length(params.LowCutoffs)
    freqNames=params.freqNames;
    currBand=freqNames(iFreqBand);
    
    %intlize
    tStart=0;
    tEnd=0;
    durations=0;
    
    %% plot PSD before filter
    srate=params.srate;
    hwelch = spectrum.welch;

    hwelch.segmentLength = params.PSD.timeWindow*srate;
    
    rawPSD=psd(hwelch,rawData,'Fs',srate);
    
    if plotFigs_flag
        rawPSDFigure=figure;set(rawPSDFigure, 'Visible', 'on');
        plot(rawPSD);
    end

    [locutoff,hicutoff]=indivdualFreq(params,iFreqBand,subj,side,elec,params.indivdualFreqband)
    
  
    hd = BandPass_butterworth_IIR_v2(srate,locutoff-1,locutoff,hicutoff,hicutoff+1);%good filter
    
    smoothdata = filter(hd,rawData');
    
    bandPSD=psd(hwelch,smoothdata,'Fs',srate);%% 
    if plotFigs_flag
        bandPSDFigure=figure;set(bandPSDFigure, 'Visible', 'off');
        plot(bandPSD)
        axesObjs = get(bandPSDFigure, 'Children');  
        dataObjs = get(axesObjs, 'Children'); 
        set(dataObjs,'Color','green');
        L = findobj(rawPSDFigure,'type','line');
        copyobj(L,findobj(bandPSDFigure,'type','axes'));
        xlim([15-locutoff hicutoff+15]);
        
    end
    
   
    %
    [yupper,ylower] = envelope(smoothdata);
    
    timeMs=1:1000/srate:(length(smoothdata)/params.srate)*1000;
    timeS=1:60/srate:(length(smoothdata)/params.srate);
    secArray=0:(1/srate):(length(smoothdata)/srate)-(1/srate); 
    if plotFigs_flag
        h3=figure; set(h3, 'Visible', 'on');
        plot(secArray,smoothdata);
        xlabel('sec');
        hold on
        plot(secArray,ylower);
        plot(secArray,yupper);
    end
    %% find the upper Percentiles
    Y = prctile(yupper,percentiles_paramter);
    percintileTh=ones(size(yupper))*Y;
    
    percentileData=yupper;
    percentileData(yupper<Y)=0;
    if plotFigs_flag
        plot(secArray,percintileTh)
        plot(secArray,percentileData)%this is the only upper Percentiles amplitude data
    end
    %% find peakes
    [pks_max,locs_max] = findpeaks(percentileData);
    findpeaks(percentileData,secArray);% plot the peaks
    
    temp2=percentileData;
    temp2(percentileData==0)=nan;
    tStart=[];
    tEnd=[];
    for iTemp2=2: length(percentileData)-2 % the value nan in this case means the vaule is under Threshold
        if (isnan(temp2(iTemp2-1))&(~isnan(temp2(iTemp2))))
            tStart=[tStart;iTemp2];
        end
        if (~isnan(temp2(iTemp2))&(isnan(temp2(iTemp2+1))))
            tEnd=[tEnd;iTemp2];
        end
        
    end
    %plot tStart and tEnd
    tStart2plot=nan(size(percentileData));
    tEnd2plot=nan(size(percentileData));
    tStart2plot(tStart)=Y;
    tEnd2plot(tEnd)=Y;
    if plotFigs_flag
        
        plot(secArray,tStart2plot,'o')
        plot(secArray,tEnd2plot,'*')
    end
    
    
    if isempty(tEnd)%
        tEnd=length(temp2);
    end
    
    
    if (tEnd(1)<tStart(1))
        tEnd(1)=[];
    end
    
    reduceStartSession=params.reduceStartSession;
    if sum(tStart<reduceStartSession)>0
        SumTstart_noSignal= sum(tStart<params.reduceStartSession);
        tStart( tStart<params.reduceStartSession)=[];
        tEnd(1:SumTstart_noSignal)=[];
    end
    
    
    if length(tStart)== length(tEnd)+1;
        if locs_max(end)>tStart(end) %cases in the last epsisoes there is no tEnd and no peak
            pks_max(end)=[];
            locs_max(end)=[];
        end
        tStart(end)=[];
    end
    if length (tEnd)~= length(tStart)
        error(' breal')
    end
    
    if sum(tStart-tEnd==0)>0 %dealing with cases tStart in the same index as tEnd
        iSamePlace=tStart-tEnd==0;
        tStart(iSamePlace)=[];
        tEnd(iSamePlace)=[];
    end
    
    
    %% find local min peaks
    %take only the highest peak if there are more than one local peak
    locs_iHighestMax=0;
    for i_tStart=1:length(tStart)
        [a,iMacLocals]=find((tStart(i_tStart)<=locs_max) &(locs_max<=tEnd(i_tStart)));
        if sum(a)>1
            [vMax,iMax]= max( pks_max(iMacLocals));
            locs_iHighestMax(i_tStart)=locs_max(iMacLocals(iMax));
        else
            locs_iHighestMax(i_tStart)=locs_max(iMacLocals);
        end
    end
    
    %plot only higest max peaskx
    maxpeaks=zeros(size(percentileData));
    maxpeaks(locs_iHighestMax)=percentileData(locs_iHighestMax);
    if plotFigs_flag
        plot(maxpeaks,'*')%
    end
    burstTimeTH=(1000/hicutoff)*2;
    burstSamplsTH=((burstTimeTH)*srate)/1000;
    list2reduce=[];
    for i=1: length(tStart)
        if  tEnd(i)-tStart(i)< burstSamplsTH
            list2reduce=[list2reduce, i];
        end
    end
    tStart(list2reduce)=[];
    tEnd(list2reduce)=[];
    locs_iHighestMax(list2reduce)=[];
    
    %
    if saveHilbert
        pathTemp=[pathHilbert,'subj',num2str(cell2mat(subj)),'_',side,'_elec',num2str(cell2mat(elec)),'_visit',num2str(visit),'_',cell2mat(OnOff),'_ipart',num2str(iPart),'\']
        path=[char(pathTemp),currBand{1},'\'];
        mkdir(path);
        fileName=[path,'hilbert']
        
        surf(peaks)
        figure(h3)
        xlim([3 8])
        set(gcf,'PaperPositionMode','auto')
        print(fileName,'-dpdf')
        print(fileName,'-depsc')
        print(fileName,'-djpeg')
        saveas(h3,[fileName,'.fig']);
        
        %plot raw data
        
        pathRAW=[char(pathTemp),'RawSignal\'];
        mkdir(pathRAW);
        fileName_raw=[pathRAW,'rawData']
        figure(rawData_fig)
        set(gcf,'PaperPositionMode','auto')
        print(pathRAW,'-dpdf')
        print(pathRAW,'-depsc')
        print(pathRAW,'-djpeg')
        saveas(rawData_fig,[pathRAW,'rawSignal.fig']);


    end
    
    %
    %
    %% building  scores featres:
    %% peak value featre
    pks_max_amp=percentileData(locs_iHighestMax);
    episodeBursts_S(iFreqBand).pks_max_amp_mean=mean(pks_max_amp);
    episodeBursts_S(iFreqBand).pks_max_amp_std=std(pks_max_amp);
    episodeBursts_S(iFreqBand).pks_max_amp_sum=sum(pks_max_amp);
    
    episodeBursts_S(iFreqBand).burstsEpsidoes_number=length(pks_max_amp);
    
    %% duration featrues
    
    durations=((tEnd-tStart)/srate).*1000;%durations in (ms)
    episodeBursts_S(iFreqBand).durations_mean=mean(durations);
    episodeBursts_S(iFreqBand).durations_std=std(durations);
    episodeBursts_S(iFreqBand).durations_sum=sum(durations);
    
    episodeBursts_S(iFreqBand).tStart_cv=(std(diff(tStart))/mean(diff(tStart)));%CV
    episodeBursts_S(iFreqBand).tEnd_cv=(std(diff(tEnd))/mean(diff(tEnd)));%CV
    
    %episode probability
    sessionLength=((length(rawData)-params.reduceStartSession)/srate);
    %     frequency=length(tStart)/sessionLength;
    trailDuration=sessionLength.*1000;%durations in (ms)
    episodeBursts_S(iFreqBand).episode_prob=sum(durations)./trailDuration;
    
    %AUC calculation;
    AUC=0;
    for i_tStart=1:length(tStart)
        Xarray=percentileData(tStart(i_tStart):tEnd(i_tStart));
        AUC(i_tStart) = trapz(Xarray);
    end
    episodeBursts_S(iFreqBand).AUC_mean=mean(AUC);
    episodeBursts_S(iFreqBand).AUC_std=std(AUC);
    episodeBursts_S(iFreqBand).AUC_sum=sum(AUC);
    
    if plotDistrubtionFiguresFlag==1
        figure(h_distrubtion);set(h_distrubtion, 'Visible', 'on');
        subplot(length(params.LowCutoffs)/2,2,iFreqBand)
        [hAx,hLine1,hLine2] = plotyy(durations,pks_max_amp,durations,AUC,'scatter','scatter');
        set(hAx(1),'ycolor','r'); hLine1.MarkerEdgeColor='r';
        set(hAx(2),'ycolor','b'); hLine2.MarkerEdgeColor='b';
        title(['Band: ',num2str(locutoff),'-',num2str(hicutoff)]);
        xlabel('Burst Duration (ms)');
        ylabel(hAx(1),'Burst amp') ;% left y-axis
        ylabel(hAx(2),'Burst AUC'); % right y-axis
    end
    
    %% moving From Struct2table
    T2(:,[char(currBand),'_episods_durations_mean'])= {episodeBursts_S(iFreqBand).durations_mean};
    T2(:,[char(currBand),'_episods_durations_std'])= {episodeBursts_S(iFreqBand).durations_std};
    T2(:,[char(currBand),'_episods_durations_sum'])= {episodeBursts_S(iFreqBand).durations_sum};
    
    T2(:,[char(currBand),'_episods_tStart_cv'])= {episodeBursts_S(iFreqBand).tStart_cv};
    T2(:,[char(currBand),'_episods_tEnd_cv'])= {episodeBursts_S(iFreqBand).tEnd_cv};
    
    T2(:,[char(currBand),'_episods_pks_max_amp_mean'])= {episodeBursts_S(iFreqBand).pks_max_amp_mean};
    T2(:,[char(currBand),'_episods_pks_max_amp_std'])= {episodeBursts_S(iFreqBand).pks_max_amp_std};
    T2(:,[char(currBand),'_episods_pks_max_amp_sum'])= {episodeBursts_S(iFreqBand).pks_max_amp_sum};
    
    T2(:,[char(currBand),'_episods_prob'])= { episodeBursts_S(iFreqBand).episode_prob};
    
    T2(:,[char(currBand),'_episods_burstsEpsidoes_number'])= {episodeBursts_S(iFreqBand).burstsEpsidoes_number};
    T2(:,[char(currBand),'_episods_AUC_mean'])= {episodeBursts_S(iFreqBand).AUC_mean};
    T2(:,[char(currBand),'_episods_AUC_std'])= {episodeBursts_S(iFreqBand).AUC_std};
    T2(:,[char(currBand),'_episods_AUC_sum'])= {episodeBursts_S(iFreqBand).AUC_sum};
    
    
    %saving the distrubtion data to a struct
    s(iFreqBand).subj=subj;
    s(iFreqBand).side=side;
    s(iFreqBand).visit=visit;
    s(iFreqBand).iPart=iPart;
    s(iFreqBand).iOnOff=OnOff;
    s(iFreqBand).elec=elec;
    s(iFreqBand).band=currBand;
    s(iFreqBand).durations=durations;
    s(iFreqBand).pks_max_amp=pks_max_amp;
    s(iFreqBand).AUC=AUC;
    s(iFreqBand).tStart=tStart;
    s(iFreqBand).tEnd=tEnd;
    s(iFreqBand).burstTimeTH=burstTimeTH;%burst minmum need to be longer than 2 cycles,time refer is in ms
    s(iFreqBand).signal={smoothdata};
    %        s(iFreqBand).envelope={h3};
    
    
    
end
%save burst distrubtion struct

filename=['BurstStruct','_subj',char(subj),'_',side,'_visit',num2str(visit),'_',char(OnOff),'_elec',char(elec{1,1}(1,1)),'_',char(elec{1,1}(1,end)),'_iPart',num2str(iPart)];
path=[params.analysisFolder,'\bursts\'];
mkdir(path);
save([path,filename,'.mat'], 's');
%toPPT
if plotDistrubtionFiguresFlag==1
    figure(h_distrubtion);set(h_distrubtion, 'Visible', 'off');
    h_distrubtion=savePicAsFig(figure(h_distrubtion));
    set(gcf,'position',get(0,'screensize'));
    toPPT(h_distrubtion,'pos','NH','Height%',200); % NEH = NorthEastHalf%  'help getPosParameters'
    toPPT({{'Subject: jur0',char(subj)},['Recording Contact Number: ',elec],{'Electrode SIDE: ',side}},'SlideNumber','current');
    
end

if (params.num_freqEpisodesNames+numberOfscores_beforeEpisodes)~=width(T2)
    error('number of scores in this section are differnt than in gen params');
end
end

