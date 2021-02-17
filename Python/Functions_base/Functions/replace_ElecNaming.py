#Daniel Sand

import pandas as pd
import numpy as np


fileName='/Tscores.csv'

newFileName='/Tscores_v3.csv'
df=pd.read_csv(fileName, sep=',')

#6 differnt electordes
oldFormat=['0-1','0-2','0-3','2-Jan','3-Jan','3-Feb']
newFormat=['0_1','0_2','0_3','2_1','3_1','3_2']

for iCont in range(0, len(oldFormat)):
    currElec_old = oldFormat[iCont]
    currElec_new = newFormat[iCont]
    df.loc[df.Elec==currElec_old,'Elec']=currElec_new
df.to_csv(path_or_buf=newFileName)
