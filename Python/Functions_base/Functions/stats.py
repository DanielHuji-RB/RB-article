#Daniel Sand

import numpy as np
import pandas as pd



def statsTable(df):

    # inlize
    columns_results = []
    df_stat = pd.DataFrame(np.nan, index=[0, ], columns=columns_results)
    line=0

    df_temp = df[df["winnerElec"] == 1]
    df_randItertions = df_temp.iloc[:, 0:].groupby( [df_temp['iRandNum']])
    for iterNum, df_iter in df_randItertions:
        df_stat=tableCreation(df_iter,df_stat,iterNum)


    return df_stat


def tableCreation(df,df_stat,iterNum):


    df_stat.loc[iterNum, 'iterNum'] = str(iterNum)
    df_stat.loc[iterNum, 'subjectNumber'] = 'Nan'
    df_stat.loc[iterNum, 'iRandNum'] = 'Nan'
    df_stat.loc[iterNum, 'featureSelection_N'] = 'Nan'


    #Train
    df_stat.loc[iterNum,'AccuracyTrain_mean'] = pd.to_numeric(df.AccuracyTrain.array).mean()
    df_stat.loc[iterNum,'AccuracyTrain_sem'] = pd.to_numeric(df.AccuracyTrain.array).std()

    df_stat.loc[iterNum, 'train_AUC_mean'] = pd.to_numeric(df.train_AUC.array).mean()
    df_stat.loc[iterNum, 'train_AUC_sem'] = pd.to_numeric(df.train_AUC.array).std()

    df_stat.loc[iterNum, 'train_Precision_mean'] = pd.to_numeric(df.train_Precision.array).mean()
    df_stat.loc[iterNum, 'train_Precision_sem'] = pd.to_numeric(df.train_Precision.array).std()

    df_stat.loc[iterNum, 'train_Recall_mean'] = pd.to_numeric(df.train_Recall.array).mean()
    df_stat.loc[iterNum, 'train_Recall_sem'] = pd.to_numeric(df.train_Recall.array).std()

    df_stat.loc[iterNum, 'trainCohen_mean'] = pd.to_numeric(df.trainCohen.array).mean()
    df_stat.loc[iterNum, 'trainCohen_sem'] = pd.to_numeric(df.trainCohen.array).std()


    #Test

    df_stat.loc[iterNum, 'testCV_accuracy_mean'] = pd.to_numeric(df.testCV_accuracy.array).mean()
    df_stat.loc[iterNum, 'testCV_accuracy_sem'] = pd.to_numeric(df.testCV_accuracy.array).std()

    df_stat.loc[iterNum, 'testCV_AUC_mean'] = pd.to_numeric(df.testCV_AUC.array).mean()
    df_stat.loc[iterNum, 'testCV_AUC_sem'] = pd.to_numeric(df.testCV_AUC.array).std()

    df_stat.loc[iterNum, 'testCV_Precision_mean'] = pd.to_numeric(df.testCV_Precision.array).mean()
    df_stat.loc[iterNum, 'testCV_Precision_sem'] = pd.to_numeric(df.testCV_Precision.array).std()

    df_stat.loc[iterNum, 'testCV_Recall_mean'] = pd.to_numeric(df.testCV_Recall.array).mean()
    df_stat.loc[iterNum, 'testCV_Recall_sem'] = pd.to_numeric(df.testCV_Recall.array).std()

    df_stat.loc[iterNum, 'testCVCohen_mean'] = pd.to_numeric(df.testCVCohen.array).mean()
    df_stat.loc[iterNum, 'testCVCohen_sem'] = pd.to_numeric(df.testCVCohen.array).std()


    #clean Validation

    df_stat.loc[iterNum, 'cleanVal_accuracy_mean'] = pd.to_numeric(df.cleanVal_accuracy.array).mean()
    df_stat.loc[iterNum, 'cleanVal_accuracy_sem'] = pd.to_numeric(df.cleanVal_accuracy.array).std()

    df_stat.loc[iterNum, 'cleanVal_AUC_mean'] = pd.to_numeric(df.cleanVal_AUC.array).mean()
    df_stat.loc[iterNum, 'cleanVal_AUC_sem'] = pd.to_numeric(df.cleanVal_AUC.array).std()

    df_stat.loc[iterNum, 'cleanVal_Precision_mean'] = pd.to_numeric(df.cleanVal_Precision.array).mean()
    df_stat.loc[iterNum, 'cleanVal_Precision_sem'] = pd.to_numeric(df.cleanVal_Precision.array).std()

    df_stat.loc[iterNum, 'cleanVal_Recall_mean'] = pd.to_numeric(df.cleanVal_Recall.array).mean()
    df_stat.loc[iterNum, 'cleanVal_Recall_sem'] = pd.to_numeric(df.cleanVal_Recall.array).std()

    df_stat.loc[iterNum, 'cleanVal_Cohen_mean'] = pd.to_numeric(df.cleanVal_Cohen.array).mean()
    df_stat.loc[iterNum, 'cleanVal_Cohen_sem'] = pd.to_numeric(df.cleanVal_Cohen.array).std()




    return df_stat



