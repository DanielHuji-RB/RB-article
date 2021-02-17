#Daniel Sand


import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout
import pandas as pd
import os
from keras.utils.vis_utils import plot_model


import sys

a=os.path.dirname(sys.executable)


path = 'C:\\Functions\\'
sys.path.insert(0,path)

# import matlab.engine
from Functions import arrangeData as AD
from Functions import DL_functions as DL_F
from sklearn.model_selection import StratifiedKFold

'''Load data:'''
localPath='/NN/'
resultPath=localPath+'/Results/'
if not os.path.exists(resultPath):
    os.makedirs(resultPath)
fileName=localPath+'/input/v2.csv'
df=pd.read_csv(fileName, sep=',')


contactFile=localPath+'/input/Contact_places.csv'
df_contcat=pd.read_csv(contactFile, sep=',')
df = pd.merge( df_contcat,df, on=['reference','Elec','side'])

print(df.shape)
df=df[df.loc[:,'Chosen_elec_SVMbased']==1].copy()
print(df.shape)


idx=df.loc[:,'labels']<1
df.loc[idx,'labels']=-1
df_intial=df.copy()

'#feature extraction'
columns_results = ['Model', 'featureSelection_N',]
df_results_new = pd.DataFrame(np.nan, index=[0, ], columns=columns_results)
line = -1

metaDataCol_N = 11
useNewData_Flag=True


df_subjects = df.iloc[:, 0:].groupby([df['reference'], df['side']])

# intlize
trainAcc_allSubj_avg = np.empty((len(df_subjects), 1,))
trainAcc_allSubj_avg[:] = np.nan
valAcc_allSubj_avg = np.empty((len(df_subjects), 1,))
valAcc_allSubj_avg[:] = np.nan
testAcc_allSubj_avg = np.empty((len(df_subjects), 1,))
testAcc_allSubj_avg[:] = np.nan
subject_number=-1

for name, df_subject in df_subjects:
    subject_number=subject_number+1
    df_train ,df_test,df_subject=AD.arrageData_orUseFixData(df_subject,metaDataCol_N,useNewData_Flag)

    x_train ,y_train,df_xTrain=AD.Seprates_Features_Labels(df_train,metaDataCol_N)
    x_test ,y_test,df_xTest=AD.Seprates_Features_Labels(df_test,metaDataCol_N)

    x_train_Orignal=x_train.copy()
    y_train_Orignal=y_train.copy()
    x_test_Orignal=x_test.copy()
    y_test_Orignal=y_test.copy()
    iter=-1

    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(df_xTrain, y_train)
    print(skf)


    # intlize
    trainAcc_avg = np.empty((skf.n_splits, 1,))
    trainAcc_avg[:] = np.nan
    valAcc_avg = np.empty((skf.n_splits, 1,))
    valAcc_avg[:] = np.nan
    testAcc_avg = np.empty((skf.n_splits, 1,))
    testAcc_avg[:] = np.nan

    trainScore_avg = np.empty((skf.n_splits, 1,))
    trainScore_avg[:] = np.nan
    valScore_avg = np.empty((skf.n_splits, 1,))
    valScore_avg[:] = np.nan
    testScore_avg = np.empty((skf.n_splits, 1,))
    testScore_avg[:] = np.nan

    modelH_Params_avg = type('', (), {})()
    modelH_Params_max = type('', (), {})()

    for train_F_index, val_F_index in skf.split(df_xTrain, y_train):

        iter = iter + 1
        if iter ==4:
            print('curr iter:' + str(iter))
        else:
            print('skip iter:'+str(iter))
            continue
        print(', '.join(str(x) for x in [name]))
        print('iter:' +str(iter) )

        print("TRAIN:", train_F_index, "TEST:", val_F_index)
        df_xTrain_F, df_xVal_F = df_xTrain.iloc[train_F_index, :], df_xTrain.iloc[val_F_index, :]
        y_train_f, y_val_f = y_train[train_F_index], y_train[val_F_index]

        x_train_f = df_xTrain_F.values.copy()
        x_val_f = df_xVal_F.values.copy()
        feature_scope=[5]#

        modelH_Params, df_results_new, line = DL_F.DL_v1(x_train_f,y_train_f,df_xTrain_F, x_val_f,y_val_f, x_test_Orignal,y_test_Orignal, feature_scope, df_results_new, line, name,iter)

        trainAcc_avg[iter]=modelH_Params.Train_accuracy
        valAcc_avg[iter]=modelH_Params.Val_accuracy
        testAcc_avg[iter]=modelH_Params.Test_accuracy

        trainScore_avg[iter] = modelH_Params.Train_score
        valScore_avg[iter] = modelH_Params.Val_score
        testScore_avg[iter] = modelH_Params.Test_score

        df_results_new.to_csv(path_or_buf=resultPath + 'DL_ModelforSubject' + (', '.join(str(x) for x in [name]))+str(iter) + '_5fold.csv')  # final

    modelH_Params_avg.Train_accuracy = trainAcc_avg.mean()
    modelH_Params_avg.Val_accuracy = valAcc_avg.mean()
    modelH_Params_avg.Test_accuracy = testAcc_avg.mean()
    modelH_Params_avg.Train_score = trainScore_avg.mean()
    modelH_Params_avg.Val_score= valScore_avg.mean()
    modelH_Params_avg.Test_score = testScore_avg.mean()

    line, df_results_new = AD.add2Excel_v2_DL(modelH_Params_avg, line,  df_results_new, name, 'mean 5fold')

    inx_iter=valAcc_avg.argmax()
    modelH_Params_max.Train_accuracy = trainAcc_avg[inx_iter][0]
    modelH_Params_max.Val_accuracy = valAcc_avg[inx_iter][0]
    modelH_Params_max.Test_accuracy = testAcc_avg[inx_iter][0]
    modelH_Params_max.Train_score = trainScore_avg[inx_iter][0]
    modelH_Params_max.Val_score = valScore_avg[inx_iter][0]
    modelH_Params_max.Test_score = testScore_avg[inx_iter][0]

    line, df_results_new = AD.add2Excel_v2_DL(modelH_Params_max, line,  df_results_new, name, 'max of 5fold')

    df_results_new.to_csv(path_or_buf=resultPath + 'DL_ModelforSubject'+ (', '.join(str(x) for x in [name]))+'_5fold.csv')
df_results_new.to_csv(path_or_buf=resultPath+'DL_5fold.csv')#final

