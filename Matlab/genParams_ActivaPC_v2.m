function params = genParams_ActivaPC_v2

params.drivePort='F';
params.saveFolder=[params.drivePort,':\dn'];
mkdir(params.saveFolder);
params.folder=[params.drivePort,':\data_v2.xlsx']
params.csvpath=fullfile(params.folder);

params.rawDataFolder=params.saveFolder;
params.processedFolder=fullfile(params.rawDataFolder,'processedTreatment');
params.interpolatedFolder=fullfile(params.processedFolder,'interpulatedTreatment');
params.cleanedFolder=fullfile(params.processedFolder,'cleanedTreatment');
params.PSDfolder=fullfile(params.processedFolder,'PSDdata');
params.subjInfoFolder=fullfile(params.processedFolder,'subjInfo');

[~,~,~]=mkdir(params.interpolatedFolder);
[~,~,~]=mkdir(params.processedFolder);
[~,~,~]=mkdir(params.cleanedFolder);
[~,~,~]=mkdir(params.PSDfolder);
[~,~,~]=mkdir(params.subjInfoFolder);
params.ClinDataFile='?.csv';
params.analysisFolder=fullfile(params.rawDataFolder,'Results');
 
params.maxIndividualFreq=['C:\LFP_analysis_v2.0','\contacts.xlsx']

params.chansLabels={'0-1','0-2','0-3','1-2','1-3','2-3'}%activa montage
params.metaDataTable=[6]%start at 1 and end at 6
%% Recording params
params.srate=422;%% change this !!!!take from XML
params.reduceStartSession=params.srate*3;
params.stds2reject = [];
params.relativePower = 1;
params.LowCutoffs =  [2 4 8 12 20 12 35 35]; %
params.HighCutoffs = [4 8 12 20 35 35 45 80]; %
params.InitialFilter_LowCutoffs=[0.5 2 3 3 3 3 30 30];%
params.InitialFilter_HiCutoffs= [6 10 15 37 37 37 90 90];%
params.notchBand_nan_flag=0;%
params.notchBand_nan=[100 102]%

params.freqNames={'delta','theta_tremor','alpha','lowBeta','highBeta','beta','lowGamma','Gamma'};
params.freqNormlizedFlag='normalized';%'non-normalized' or 'normalized'
params.pnina_normlizeFlag=0;
params.indivdualFreqband=0;%this flag represent if the freq band are calculte induvidual or for all subj the same. 1 is individual 0 freq cutoff

params.zScoreFlag=1;% 
params.Range2CalculateFrom = [1 80]; % 
params.flagShortenedPSD=1;% in order to save space and upload time; 0=all freqs; 1= save freqs from 0 to 200Hz
%table indexs
params.num_freqNames=length(params.freqNames);
params.num_freqRatioNames=nchoosek(length(params.freqNames),2);
params.num_freqEpisodesNames=13*length(params.freqNames);%13 is the number of featers in the table
%% This is because a high res will create too many peaks in the PSD and may lead to wrong peak detection
params.peakThreshold = 0.4; % 
params.relORabs = 1; % 
params.forcePeak = 0; 
params.extendMargines = 1; %

params.PSD.timeWindow=2;% in [seconds] 
params.PSD.timeOverlap=0%
%% analysis
params.analysis.subjNames2fit={'1','3','5','6'};
params.analysis.subjElecs={{'2-3';'2-3'},{'2-3';'2-3'},{'1-2';'1-2'},{'2-3';'2-3'}};%each electrode for each side (Left & Right) Respectively for the {left; right}

params.analysis.RelventElec_Flag=0;% 1=run only the relevent elec each subject has its own specfic electrode that place in the motor STN part
params.analysis.subjNames2test={};
params.analysis.remisssubjNames={};

%%flags
params.flag.splitData_Flag=0;
params.replaceNotchWithNan_flaf=1;


%% fixes
%% simulated Data need to be mark if not in use
  params.simulated_flag=0;

%% ****ens simulated params
end