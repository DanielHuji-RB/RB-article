function [MS_between MS_within]=ANOVA_MS_with_missing_values(dataMatrix)

%dataMatrix - visits*subjects
%missing data is nan


% dataMatrix=[
%     0.0571	0.0873	0.0974	0.1033	0.0703;
%     0.0813	0.0662	0.1352	0.0915	0.1026;
% 	0.0831	0.0672	0.0817	0.0781	0.0956;
%     0.0976	0.0819	0.1016	0.0685	0.0973;
%     0.0817	0.0749	0.0968	0.0677	0.1039;
%     0.0859	0.0649	0.1064	0.0697	0.1045;
%     0.0735	0.0835	0.105	0.0764	nan;
%     0.0659	0.0725	nan     0.0689	nan;
%     0.0923	nan		nan     nan     nan;
%     0.0836	nan		nan     nan     nan;
% ];
% dataMatrix=[79.65091195	96.88473506	74.20023105	97.5018815	nan	63.71768037	60.61681726	42.78152256	44.04964382	80.19861325	90.25722016	91.32778977	75.87180367	nan	76.74775904	97.01218394	95.5806172	37.63982299	98.67560574;
% 70.96408647	90.66318602	96.38566597	95.22569471	14.67498259	92.92249978	40.14310822	42.16783723	53.22961401	96.53806614	21.24750761	88.9712656	62.15841666	91.38246311	87.81045886	98.85336819	90.86811335	92.88794696	80.84942212;
% 46.66431302	78.36020935	85.79168789	62.75204051	3.685453075	87.23589482	18.41985986	72.5653629	65.07198971	84.3936866	42.39154648	98.45536058	81.06220496	93.39718142	nan	96.63138958	nan	59.0123533	96.33632233
% ];

numberOfvisits=size(dataMatrix,1);
numberOfGroups=size(dataMatrix,2);
grandMean=0;
for iGroup=1:numberOfGroups
    notNanValues=~isnan(dataMatrix(:,iGroup));
    numberOfVisitsVec(iGroup)=sum(notNanValues);
    meanInGroup(iGroup) = mean(dataMatrix(notNanValues,iGroup),1);
    grandMean = grandMean +  sum(dataMatrix(notNanValues,iGroup),1);
end

DF_between = numberOfGroups-1;
DF_withhin = sum(numberOfVisitsVec)-numberOfGroups;
N = sum(numberOfVisitsVec);

grandMean = grandMean/N;

for iGroup=1:numberOfGroups
    groupValues=dataMatrix(~isnan(dataMatrix(:,iGroup)),iGroup);
    sumOfSquersInGroup(iGroup) = sum((groupValues-mean(groupValues)).^2);
    sumOfSquersAcross(iGroup) = numberOfVisitsVec(iGroup)*((meanInGroup(iGroup)-grandMean)^2);
end
% 
SS_between = sum(sumOfSquersAcross);
SS_within = sum(sumOfSquersInGroup);

MS_between = SS_between/DF_between;
MS_within = SS_within/DF_withhin;
