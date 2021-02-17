#Daniel Sand


import numpy as np
import pandas as pd
import os
import matlab.engine
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from Functions import arrangeData as AD
from Functions import corrBetweenFeatures as CF
from Functions import FeaturesSelection_script as FSS
from Functions import SVMs_function as SVM_F
from Functions import stats
#from Functions import count_num_feaures as CNF
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve,roc_auc_score, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix,cohen_kappa_score,balanced_accuracy_score
from sklearn.svm import SVC
import math



def manyScores_andThe_score(y_true, y_pred):


    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    score=accuracy_score(y_true, y_pred)
    return score


'''Load data:'''
randFeature_flag=False
top8Global_fea_flag=True
top7Global_fea_flag=False
Group_flag= True#

if Group_flag:
    resultPath='/Users/Group/'
else:
    resultPath='/Users/Individual/'


if not os.path.exists(resultPath):
     os.makedirs(resultPath)

localPath='/SVM/'
fileName=localPath+ 'input/Tscores.csv'
df=pd.read_csv(fileName, sep=',')

contactFile=localPath+'/Contact_places.csv'
df_contcat=pd.read_csv(contactFile, sep=',')
df = pd.merge( df_contcat,df, on=['reference','Elec','side'])


idx=df.loc[:,'labels']<1
df.loc[idx,'labels']=-1
metaDataCol_N = 11
useNewData_Flag=True

columns_results = []
df_results_new = pd.DataFrame(np.nan, index=[0, ], columns=columns_results)
line = -1

if Group_flag:
    df_subjects = df.iloc[:, 0:].groupby([df['iVisitType']])
else:
    df_subjects = df.iloc[:, 0:].groupby([df['reference'], df['side'],df['Elec']])

subject_number=-1
eng = matlab.engine.start_matlab()
iter=-1

if randFeature_flag:
    randIter_number = 1000
else:
    randIter_number = 1
for iRandNum in range(0, randIter_number):


    for name, df_subject in df_subjects:
        subject_number=subject_number+1
        df_train ,df_cleanVal,df_subject=AD.arrageData_orUseFixData(df_subject,metaDataCol_N,useNewData_Flag)
        x_train ,y_train,df_xTrain=AD.Seprates_Features_Labels(df_train,metaDataCol_N)
        x_cleanVal ,y_cleanVal,df_xcleanVal=AD.Seprates_Features_Labels(df_cleanVal,metaDataCol_N)

        x_train_Orignal=x_train.copy()
        y_train_Orignal=y_train.copy()
        x_cleanVal_Orignal=x_cleanVal.copy()
        y_cleanVal_Orignal=y_cleanVal.copy()

        feature_scope=[5]#
        # top8feaName = ['beta_episods_AUC_sum', 'theta_tremor_highBeta_Ratio_power_MeanPSD',
        #                'lowBeta_beta_Ratio_power_MeanPSD', 'highBeta_Gamma_Ratio_power_MeanPSD',
        #                'theta_tremor_lowBeta_Ratio_power_MeanPSD', 'lowBeta_power_MeanPSD',
        #                'Gamma_episods_AUC_sum', 'lowGamma_episods_AUC_sum']

        for iFeature in range(0, len(feature_scope)):

                # Feature Selection
                featureNum = feature_scope[iFeature]
                iter=iter+1

                # feature selection
                if randFeature_flag:
                    fea = [np.random.randint(140, size=featureNum)][0]
                    fea_name=df_xTrain.columns[fea].tolist()

                elif top8Global_fea_flag:
                        fea=np.empty( len(top8feaName), dtype=int)
                        for j in range(0, len(top8feaName)):
                            print(df_xTrain.columns.get_loc(top8feaName[j]))
                            fea[j]   = df_xTrain.columns.get_loc(top8feaName[j])
                        fea_name = top8feaName

                elif top7Global_fea_flag:
                        top7feaName= top8feaName.copy()
                        del top7feaName[math.floor(subject_number/6)]

                        fea=np.empty( len(top7feaName), dtype=int)
                        for j in range(0, len(top7feaName)):
                            print(df_xTrain.columns.get_loc(top7feaName[j]))
                            fea[j]   = df_xTrain.columns.get_loc(top7feaName[j])
                        fea_name = top7feaName

                else:
                    fea, fea_name = FSS.FS_MRMR(x_train_Orignal, y_train_Orignal, featureNum, df_xTrain, eng)


                x_train = x_train_Orignal[:, fea].copy()
                x_cleanVal = x_cleanVal_Orignal[:, fea].copy()

                tuned_parameters = [
                    {'kernel': ['linear'], 'C': [0.000001, 0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}]

                model = SVC()

                scorer = {
                    'roc_auc_score': make_scorer(roc_auc_score),
                    'precision_score': make_scorer(precision_score),
                    'recall_score': make_scorer(recall_score),
                    'cohen_kappa_score':make_scorer(cohen_kappa_score),
                    'balanced_accuracy_score':make_scorer(balanced_accuracy_score),
                    'specificity': make_scorer(recall_score, pos_label=-1),
                    'accuracy_score': make_scorer(accuracy_score)
                }


                clf = GridSearchCV(model, tuned_parameters, cv=5, scoring=scorer, n_jobs=-1, iid=False ,refit='accuracy_score',return_train_score=True)
                clf.fit(x_train, y_train.ravel())
                if not randFeature_flag:
                    AD.plot_coefficients_v2(clf, fea_name,resultPath,iter,str(name))
                    AD.plot_abs_coefficients_v2(clf, fea_name,resultPath,iter,str(name))
                    fea_name_str = listToStr = ' '.join([str(elem) for elem in fea_name])
                    CF.corrmatrix_features_abs(df_train.loc[:,fea_name], 0, resultPath, iter,name)
                    CF.covMatrix_features_abs(df_train.loc[:,fea_name], 0, resultPath, iter,name)

                [psd_num,burst_num]=AD.count_psd_vs_burst_number(fea_name)

                print('current Subject: ' + str(name))

                def stringValue(datesetType,metricName,clf,stdFlag=True):

                    if stdFlag:
                        sValue = [ str(round(clf.cv_results_['mean_'+datesetType+'_'+metricName+'_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_train_accuracy_score'][clf.best_index_], 2)) + ')']
                    else:
                        sValue= [ str(round(clf.cv_results_['mean_'+datesetType+'_'+metricName+'_score'][clf.best_index_], 2)) ]
                    return sValue

                def stringValue_noScore(datesetType,metricName,clf,stdFlag=True):

                    if stdFlag:
                        sValue = [ str(round(clf.cv_results_['mean_'+datesetType+'_'+metricName][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_train_accuracy_score'][clf.best_index_], 2)) + ')']
                    else:
                        sValue= [ str(round(clf.cv_results_['mean_'+datesetType+'_'+metricName][clf.best_index_], 2)) ]
                    return sValue

                # '''Results''':
                #Train
                stdFlag= False

                trainAccuracy  = stringValue( 'train', 'accuracy', clf, stdFlag )
                trainAUC       = stringValue( 'train', 'roc_auc', clf, stdFlag )
                trainCohen     = stringValue( 'train', 'cohen_kappa', clf, stdFlag )
                trainPrecision = stringValue( 'train', 'precision', clf, stdFlag )
                trainRecall    = stringValue( 'train', 'recall', clf, stdFlag )
                trainSpecificity    = stringValue_noScore( 'train', 'specificity', clf, stdFlag )

                #Test CV:

                testCVAccuracy = stringValue('test', 'accuracy', clf, stdFlag)
                testCVAUC = stringValue('test', 'roc_auc', clf, stdFlag)
                testCohen = stringValue('test', 'cohen_kappa', clf, stdFlag)
                testPrecision = stringValue('test', 'precision', clf, stdFlag)
                testRecall = stringValue('test', 'recall', clf, stdFlag)
                testCVSpecificity = stringValue_noScore('test', 'specificity', clf, stdFlag)

                #Valditon Clean Data:
                y_cleanVal_pred = clf.predict(x_cleanVal)
                print()

                cleanValAccuracy =accuracy_score(y_cleanVal.ravel(), y_cleanVal_pred)
                cleanVal_roc_auc=roc_auc_score(y_cleanVal.ravel(),y_cleanVal_pred)
                cleanVal_Cohen=cohen_kappa_score(y_cleanVal.ravel(),y_cleanVal_pred)
                cleanVal_precision=precision_score(y_cleanVal.ravel(),y_cleanVal_pred)
                cleanVal_recall=recall_score(y_cleanVal.ravel(),y_cleanVal_pred)
                cleanVal_specificity=recall_score(y_cleanVal.ravel(),y_cleanVal_pred,pos_label=-1)


                line, df_results_new = AD.add2Excel_v4_gridSearch(iRandNum,psd_num,burst_num, trainAccuracy,trainAUC,trainPrecision,trainRecall,trainCohen,trainSpecificity,testCVAccuracy,testCVAUC,testPrecision,testRecall,testCohen,testCVSpecificity,cleanValAccuracy,cleanVal_roc_auc,cleanVal_precision,cleanVal_recall,cleanVal_Cohen,cleanVal_specificity,featureNum,clf.best_params_, line, df_results_new, name,math.floor(subject_number/6),'mean CV 5fold- ACC & AUC ')
                if name[2]=='3_2':
                    arr_allelecAcc=df_results_new.testCV_accuracy[line-5:line+1]
                    indHigestAcc=arr_allelecAcc.values.argmax()
                    winnerElecInd=(line - 6 + indHigestAcc)
                    df_results_new.loc[winnerElecInd, 'winnerElec'] = 1
    subject_number=0

df_results_new.to_csv(path_or_buf=resultPath+'SVM_modelPerAllCVfolds.csv')

df_stats=stats.statsTable(df_results_new)
df_stats.to_csv(path_or_buf=resultPath+'SVM_average_onAll_Iterations.csv')

