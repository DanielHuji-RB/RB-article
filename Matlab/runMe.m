%Daniel Sand
function runMe
clear all
close all
clc
%% Global Var
restoredefaultpath
[allCodesPath,~,~]=fileparts(mfilename('fullpath')); % Get the path to file location
cd(allCodesPath);                                 % Set the file path as current folder
addpath(genpath(cd));                                  % Add subfolders to path
a=fileparts(allCodesPath);
b=fileparts(a);
addpath(genpath(a));
addpath(genpath([a,'\PPT\toPPT']));
addpath(genpath('C:\Codes\'));%


params=genParams_ActivaPC_v2;

allVisitFlag=1;%
rerunFlag=1% 
winTime4SplitingData=5;%
TsoreName=['z^2_Whitening_plusNewFilters_',num2str(winTime4SplitingData),'sec'];%
if winTime4SplitingData>27
    splitData_Flag=0;%
else
    splitData_Flag=1;%
end

    %% take out data from Excel
    Tscores=table;
    T_NotchFilter=table;
    T= readtable(params.csvpath);
    experimentDay_temp=(char(T.FileName(:,1)));
    experimentDay=experimentDay_temp(:,7:16);
    T.Day=cellstr(experimentDay);
    T=T(T.isMontage==1,:);% take only montage tasks
    T=T(T.TempFlag==1,:);% take only montage tasks
    
    
    %% run over all files
    
    subjList=params.analysis.subjNames2fit;
    for iSubjL=1:length(subjList)
        
        iSubj=subjList(iSubjL)
        T_subj=T(strcmp( T.Patient,{['jur0',char(iSubj)]}),:);
        MontageType=unique(T_subj.Task);
        for iMontage=1
            T_montage=T_subj(strcmp( T_subj.Task,(MontageType(iMontage))),:);
          
            LFP_Elec=params.chansLabels;
            
            for iElec=1:length(LFP_Elec)
                
                if  params.analysis.RelventElec_Flag
                    if ~strcmp(LFP_Elec(iElec),unique(params.analysis.subjElecs{1,iSubjL}))
                        continue
                    end
                end
                PSD_visit_R=figure;grid on; set(PSD_visit_R, 'Visible', 'off');
                PSD_visit_L=figure;grid on;set(PSD_visit_L, 'Visible', 'off');
                
                T_elec=T_montage(strcmp(T_montage.Ch1,LFP_Elec(iElec)),:);
                onOfftype={'OFF','ON'};
                for iOnOff=1:length(onOfftype)

                    iOnOff
                    T_Off=T_elec(strcmp(T_elec.Med,onOfftype(iOnOff)),:);
                    days=sort(unique(T_Off.Day));
                    firstVisit_flag=1;
                    for iVisit=1:length(days)
                        iVisit
                        T_day= T_Off(strcmp( T_Off.Day,days(iVisit)),:);
                        %% loadata
                        specficFile_temp=char(T_day.DayFolder);
                        specficFile=[params.drivePort,':\data_from_pnina',specficFile_temp(3:end),'\',char(T_day.FileName)];
                        if params.simulated_flag
                            chanT_temp=load(specficFile)
                            chanT=array2table(chanT_temp.bothSide)
                        else
                            chanT=readtable(specficFile);
                            [ xmlStruct ] = xml2struct( [specficFile,'.xml'] )
                            
                        end
                 
                        %remove beging montage noise
                        srate=params.srate;%% 
                        chanT.Var1(1:params.reduceStartSession)=0; 
                        chanT.Var3(1:params.reduceStartSession)=0;
                        
                        %% cleaning_procedure
                        filtern=4; BandFreq=[0.5 100]; ftype='bandpass';    
                        NeqFreq=srate/2;
                        Wn=BandFreq/NeqFreq;                               
                        [b, a] = butter(filtern,Wn,ftype);
                        chanT.Var1 = filtfilt(b,a,chanT.Var1);
                        chanT.Var3 = filtfilt(b,a,chanT.Var3); 
                        
                        %remove AKG artifact
                        isZero=char(LFP_Elec(iElec));
                        if  ismember(str2num(char(iSubj)),[1:6]) & isZero(1)=='0'
                            if (iSubjL==4)& (iVisit==5) 
                            else
                                [x_raw_old, temp1,CV] = templatesubtract_v3(chanT.Var1,srate,'interpolate','off');
                                chanT.Var1=temp1';
                            end
                        end
                        %Left
                        if  ismember(str2num(char(iSubj)),[3,6])& isZero(1)=='0'
                            [x_raw_old, temp3,CV] = templatesubtract_v3(chanT.Var3,srate,'interpolate','off');
                            chanT.Var3=temp3';
                        end
                        
                        
                    
                        Run_notchflag=1
                        if Run_notchflag
                            notch_freq1=findNotchPeak(chanT.Var1,params,45,60)
                            chanT.Var1 =notchFromGoogle(chanT.Var1,srate,notch_freq1);
                            notch_freq3=findNotchPeak(chanT.Var3,params,45,60)
                            chanT.Var3 =notchFromGoogle(chanT.Var3,srate,notch_freq3);
                            T_day.notch_freq1=notch_freq1
                            T_day.notch_freq3=notch_freq3

                            T_NotchFilter=[T_NotchFilter;T_day]; 


                            %notch to remove clock of the montage
                            notch_freq=findNotchPeak(chanT.Var1,params,100,120)
                            chanT.Var1 =notch105Hz_v2(chanT.Var1,srate,notch_freq-2,notch_freq+2);
                            notch_freq=findNotchPeak(chanT.Var3,params,100,120)
                            chanT.Var3 =notch105Hz_v2(chanT.Var3,srate,notch_freq-2,notch_freq+2);
                        end
                        
                        
                        
                        if params.zScoreFlag
                            %Var 1
                            chanT_z=table;
                            nonZeoro_chanT.Var1=chanT.Var1(params.reduceStartSession:end);
                            zVar1_temp=zscore_func(nonZeoro_chanT.Var1);
                            chanT_z.Var1=[zeros(params.reduceStartSession,1);zVar1_temp];
                            %Var 3
                            nonZeoro_chanT.Var3=chanT.Var3(params.reduceStartSession:end);
                            zVar3_temp=zscore_func(nonZeoro_chanT.Var3);
                            chanT_z.Var3=[zeros(params.reduceStartSession,1);zVar3_temp];
                            
                        else
                            chanT_z=table;
                            chanT_z.Var1= chanT.Var1;
                            chanT_z.Var3= chanT.Var3;
                        end
                        %% analysis
                      
                        
                        splitToTrainAndTest_Flag=1
                        if splitData_Flag==1
                            overLapWin=0.5;%
                            nonZeoro_chanT_temp=chanT(params.reduceStartSession:end,:);
                            
                            if splitToTrainAndTest_Flag==0
                                [splitedData,~]=split_RB_Datachan(nonZeoro_chanT_temp,srate,winTime4SplitingData,overLapWin,'NaN');
                            else
                                trainPercent=0.68; %
                                
                                %Train
                                nonZeoro_chanT_train=nonZeoro_chanT_temp(1:height(nonZeoro_chanT_temp)*trainPercent,:);%Train Data
                                [splitedData_train,lastIndex]=split_RB_Datachan(nonZeoro_chanT_train,srate,winTime4SplitingData,overLapWin,'Train');
                                %Test
                                nonZeoro_chanT_test=nonZeoro_chanT_temp(lastIndex:end,:);%Test
                                [splitedData_test,~]=split_RB_Datachan(nonZeoro_chanT_test,srate,winTime4SplitingData,overLapWin,'Test');
                                if length(splitedData_test)*4.1< length(splitedData_train)%*4.1 means that is over than 80%
                                    warning('your Traing set is bigger then 80% of the data, consider to reduce trainPercent value or recuce the winTime4SplitingData'  );
                                end
                                
                                splitedData=[splitedData_train;splitedData_test];
                                
                            end
                        else
                            splitedData=1
                            iPartLabel='union'
                        end
                        
                        for iPart=1:length(splitedData)
                            if splitData_Flag==1
                                clear var chanT_z
                                chanT_z=table
                                chanT_z.Var1= splitedData{iPart,1}.Var1
                                chanT_z.Var3= splitedData{iPart,1}.Var3
                                TrainOrTest_label= splitedData{iPart,2}
                            else
                                TrainOrTest_label= 'nonSlpitData';
                            end
                            
                            %Right side
                            side={'right'}
                            if  params.analysis.RelventElec_Flag
                                if strcmp(LFP_Elec(iElec),(params.analysis.subjElecs{1,iSubjL}(2,1)))%2 refer to Right
                                    [spectogram_fig,psd_figR,Tscores1]=LFPanalysis(chanT_z.Var1,params,char(side),iSubj,LFP_Elec(iElec),iVisit,onOfftype(iOnOff),iPart);%calculate PSD
                                    Tscores=TableBuilder(Tscores,Tscores1,iSubj,side,iVisit,onOfftype(iOnOff),LFP_Elec(iElec),MontageType(iMontage),iPart,TrainOrTest_label,xmlStruct);
                                    if strcmp(onOfftype(iOnOff),'OFF')
                                        PSD_visit_R=addPSD2Fig(params,PSD_visit_R,psd_figR,'b')
                                    else if strcmp(onOfftype(iOnOff),'ON')
                                            PSD_visit_R=addPSD2Fig(params,PSD_visit_R,psd_figR,'r')
                                        end
                                    end
                                end
                            else 
                                
                                [spectogram_fig,psd_figR,Tscores1]=LFPanalysis(chanT_z.Var1,params,char(side),iSubj,LFP_Elec(iElec),iVisit,onOfftype(iOnOff),iPart);%calculate PSD
                                Tscores=TableBuilder(Tscores,Tscores1,iSubj,side,iVisit,onOfftype(iOnOff),LFP_Elec(iElec),MontageType(iMontage),iPart,TrainOrTest_label,xmlStruct);
                                if strcmp(onOfftype(iOnOff),'OFF')
                                    PSD_visit_R=addPSD2Fig(params,PSD_visit_R,psd_figR,'b')
                                else if strcmp(onOfftype(iOnOff),'ON')
                                        PSD_visit_R=addPSD2Fig(params,PSD_visit_R,psd_figR,'r')
                                    end
                                end
                                % end
                            end
                            
                            
                            %Left side
                            side={'left'}
                            if params.analysis.RelventElec_Flag
                                if strcmp(LFP_Elec(iElec),(params.analysis.subjElecs{1,iSubjL}(1,1)))%1 refer to Left
                                    maxFreqValues_Left(iVisit,iOnOff,iSubjL,:)=findFreqPeak(chanT_z.Var3,params,params.freqNormlizedFlag);
                                    
                                    [spectogram_fig,psd_figL,Tscores1]=LFPanalysis(chanT_z.Var3,params,char(side),iSubj,LFP_Elec(iElec),iVisit,onOfftype(iOnOff),iPart);%calculate PSD
                                    Tscores=TableBuilder(Tscores,Tscores1,iSubj,side,iVisit,onOfftype(iOnOff),LFP_Elec(iElec),MontageType(iMontage),iPart,TrainOrTest_label,xmlStruct);
                                    if strcmp(onOfftype(iOnOff),'OFF')
                                        PSD_visit_L=addPSD2Fig(params,PSD_visit_L,psd_figL,'b')
                                    else if strcmp(onOfftype(iOnOff),'ON')
                                            PSD_visit_L=addPSD2Fig(params,PSD_visit_L,psd_figL,'r')
                                        end
                                    end
                                end
                            else  
                                [spectogram_fig,psd_figL,Tscores1]=LFPanalysis(chanT_z.Var3,params,char(side),iSubj,LFP_Elec(iElec),iVisit,onOfftype(iOnOff),iPart);%calculate PSD
                                Tscores=TableBuilder(Tscores,Tscores1,iSubj,side,iVisit,onOfftype(iOnOff),LFP_Elec(iElec),MontageType(iMontage),iPart,TrainOrTest_label,xmlStruct);
                                if strcmp(onOfftype(iOnOff),'OFF')
                                    PSD_visit_L=addPSD2Fig(params,PSD_visit_L,psd_figL,'b')
                                else if strcmp(onOfftype(iOnOff),'ON')
                                        PSD_visit_L=addPSD2Fig(params,PSD_visit_L,psd_figL,'r')
                                    end
                                end
                            end
                        end% iPart
                        
                        if iVisit==1 
                            firstVisit_flag=firstVisit_flag+1;
                            T_alldays = chanT_z;
                            dayIndex(iVisit)=height(chanT_z);
                            dayNumber=1;
                        else
                            T_alldays = [T_alldays;chanT_z];
                            dayIndex(iVisit)= dayIndex(iVisit-1)+height(chanT_z);
                            dayNumber=dayNumber+1;
                        end
                        
                        
                    end %iVisit
                    %% analysis menu
                    if strcmp(onOfftype(iOnOff),'OFF')
                        T_alldays_off=T_alldays;
                    else if strcmp(onOfftype(iOnOff),'ON')
                            T_alldays_on=T_alldays;
                        else
                            error(' the med type is not OFF or ON')
                            break
                        end
                    end
                end
                
                
                if allVisitFlag
                    %right side
                    if  params.analysis.RelventElec_Flag
                        if strcmp(LFP_Elec(iElec),(params.analysis.subjElecs{1,iSubjL}(2,1)))%2 refer to Right
                            [Tscores]=plotOFF_On2ppt(T_alldays_off.Var1,T_alldays_on.Var1,iSubj,MontageType(iMontage),LFP_Elec(iElec),iElec,params,'right',Tscores,PSD_visit_R,iPart,TrainOrTest_label,xmlStruct)
                        end
                    end
                    %left side
                    if params.analysis.RelventElec_Flag
                        if 1
                            [Tscores]=plotOFF_On2ppt(T_alldays_off.Var3,T_alldays_on.Var3,iSubj,MontageType(iMontage),LFP_Elec(iElec),iElec,params,'left',Tscores,PSD_visit_L,iPart,TrainOrTest_label,xmlStruct)
                        end
                    end
                end
                close all
            end
        end
    end
end


        
        
        %save
        folder=[params.subjInfoFolder,'\winSize_',num2str(winTime4SplitingData),'sec']
        mkdir(folder)
        save(fullfile(folder,['subj-',char(iSubj),TsoreName,'_Tscores.mat']),'Tscores','-v7.3');
        
        %     save(fullfile(params.subjInfoFolder,'subjs_table.mat'),'subjs_table','-v7.3');
        if splitToTrainAndTest_Flag==1
            TrainNum= sum(strcmp(Tscores.TrainOrTest,'Train'))
            TestNum= sum(strcmp(Tscores.TrainOrTest,'Test'))
            trainPercent=(TrainNum/(TestNum+TrainNum))*100
            testPercent=(TestNum/(TestNum+TrainNum))*100
            if testPercent<20
                error('if your Traing set is bigger then 80% of the data, consider to reduce trainPercent value or recuce the winTime4SplitingData'  );
                break
            end
        end
    end
    save(fullfile(folder,['allSubjs',TsoreName,'_Tscores.mat']),'Tscores','-v7.3');
    writetable(Tscores,fullfile(folder,['allSubjs',TsoreName,'_Tscores.csv']))
    
    
end %end reRun
testTypeName='Montage_475'

