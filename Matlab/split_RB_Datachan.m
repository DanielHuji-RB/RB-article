%Daniel Sand

function [splitedData,lastIndex]=split_RB_Datachan(data,srate,time,overlapWin,TrainOrTest)%
winNumber=floor(height(data)/(srate*time));
dataSizePerAbsWin=(height(data)/(srate));
winSize=srate*time;
winNum_overlap=floor((height(data)/srate)-(overlapWin)+1);
numOfSplitData=floor(((dataSizePerAbsWin-time)/overlapWin)+1);

if overlapWin==0
    for iPart=0:winNumber-2
        splitedData{iPart+1,1}=data(winSize*iPart+1:winSize*iPart+winSize,:,TrainOrTest)
        splitedData{iPart+1,2}=TrainOrTest;

    end
    splitedData{iPart+2,1}=data(winSize*(iPart+1):end,:)
    splitedData{iPart+1,2}=TrainOrTest;
        lastIndex=height(data);


else
    for iPart=0:numOfSplitData-1 
        iPart
        step=iPart*srate*overlapWin;
        splitedData{iPart+1,1}=data(step+1:step+winSize,:);
        splitedData{iPart+1,2}=TrainOrTest;
        arr1(iPart+1)=step+1;
        arr2(iPart+1)=step+winSize;
    end
    lastIndex=step+winSize;

    
    
end
end