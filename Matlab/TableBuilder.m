%Daniel Sand



function Tscores=TableBuilder(Tscores,Tscores1,iSubj,side,iVisit,OnOff_value,elec,iVisitType,iPart,TrainOrTest_label,xmlStruct)


tblB=table;
tblB.reference=iSubj;
tblB.side= side;
tblB.iVisit={iVisit};
tblB.iPart={iPart};
tblB.TrainOrTest={TrainOrTest_label};
if strcmp(OnOff_value,'OFF') | strcmp(OnOff_value,'Off')
    tblB.labels=0;
end
if strcmp(OnOff_value,'ON')|strcmp(OnOff_value,'On')
    tblB.labels=1;
end
tblB.OnOff=OnOff_value;
tblB.Elec=elec;
tblB.iVisitType=iVisitType;

XML_flag=0
if XML_flag
    tblB.patientID=  {xmlStruct.RecordingItem.PatientID.Text};
    tblB.date={xmlStruct.RecordingItem.INSTimeStamp.Text};
    tblB.RecordingDuration= {xmlStruct.RecordingItem.RecordingDuration.Text};
    tblB.sampleRate={ xmlStruct.RecordingItem.SenseChannelConfig.TDSampleRate.Text};
    
    if  strcmp(side,'right')
        CH='1';
    else
        CH='3';
    end
    tblB.creFreq={eval(['xmlStruct.RecordingItem.SenseChannelConfig.Channel',CH,'.CtrFreq.Text'])};
    tblB.PowGain={eval(['xmlStruct.RecordingItem.SenseChannelConfig.Channel',CH,'.PowGain(1,1).Text'])};
    tblB.TDGain={eval(['xmlStruct.RecordingItem.SenseChannelConfig.Channel',CH,'.TDGain(1,1).Text'])};
    tblB.PlusInput={eval(['xmlStruct.RecordingItem.SenseChannelConfig.Channel',CH,'.PlusInput(1,1).Text'])};
    tblB.MinusInput={eval(['xmlStruct.RecordingItem.SenseChannelConfig.Channel',CH,'.MinusInput(1,1).Text'])};
    tblB.TimeGainActuals_250={(eval(['xmlStruct.RecordingItem.SenseChannelConfig.Channel',CH,'.TimeGainActuals.Gain250.Text']))};
    tblB.PowerGainActuals_500={(eval(['xmlStruct.RecordingItem.SenseChannelConfig.Channel',CH,'.PowerGainActuals.BW5Hz.Gain500.Text']))};
end


Tscores1 = [tblB Tscores1];
if height(Tscores)==0
    Tscores=Tscores1;
else
    Tscores=[Tscores; Tscores1];
end

end