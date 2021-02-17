#Daniel Sand
#31/1/2019/
#updated on 14/12/2020

import numpy as np
import pandas as pd
import os
import matlab.engine
from Functions import arrangeData as AD
from Functions import corrBetweenFeatures as CF
from Functions import FeaturesSelection_script as FSS
from Functions import SVMs_function as SVM_F
from Functions import count_num_feaures as CNF
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve,roc_auc_score, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix,cohen_kappa_score,balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import math
from sklearn.dummy import DummyClassifier


def manyScores_andThe_score(y_true, y_pred):
    #this function calculte relvent scores such as AUC Senstivuty specfity recall etc' but only one score will be the one to use for the right grid paramters..


    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('auc:'+str(auc))

    # Recall
    print(classification_report(y_true, y_pred))

    sensitivity,specificity = SVM_F.confusion_matrix_func(y_true, y_pred)
    print('sensitivity:' + str(sensitivity) + '; specificity:' + str(specificity))

    score=accuracy_score(y_true, y_pred)

    return score

Group_flag=False #True for the group model, False for individual model

'''Load data:'''
if Group_flag:
    resultPath='/2020/Group/Dec2020_group_5Fearures_forMRMR_5fold_V3/'
else:
    resultPath='/2020/Individual/Dec2020_idividualModel_5Feature_forMRMR_5fold_V3/'

if not os.path.exists(resultPath):
     os.makedirs(resultPath)

localPath='/Codes/PycharmProjects/SVM/'

fileName=localPath+ '/Tscores_v3.csv'
df=pd.read_csv(fileName, sep=',')

# add Contact Place -DLOR VMNR Zona Incerta
contactFile=localPath+'contactPlaces.csv'
df_contcat=pd.read_csv(contactFile, sep=',')
df = pd.merge( df_contcat,df, on=['reference','Elec','side'])

idx=df.loc[:,'labels']<1 #changeing the lables 0 to be -1
df.loc[idx,'labels']=-1 #changeing the lables 0 to be -1


metaDataCol_N = 11
useNewData_Flag=True

'#feature extraction'

columns_results = []  # inlize df results
df_results_new = pd.DataFrame(np.nan, index=[0, ], columns=columns_results)  # inlize df results
line = -1


if Group_flag:
    df_subjects = df.iloc[:, 0:].groupby([df['iVisitType']])
else:
    df_subjects = df.iloc[:, 0:].groupby([df['reference'], df['side'],df['Elec']])


# intlize
subject_number=-1
eng = matlab.engine.start_matlab()
iter=-1

'''Loop on all Subjects and Electrodes'''
for name, df_subject in df_subjects:
    subject_number=subject_number+1
    '''arrange Data: outlaiers normlize and Split to train and cleanVal:'''
    df_train ,df_cleanVal,df_subject=AD.arrageData_orUseFixData(df_subject,metaDataCol_N,useNewData_Flag)
    'seprate the table to features and labels:'
    x_train ,y_train,df_xTrain=AD.Seprates_Features_Labels(df_train,metaDataCol_N)
    x_cleanVal ,y_cleanVal,df_xcleanVal=AD.Seprates_Features_Labels(df_cleanVal,metaDataCol_N)

    x_train_Orignal=x_train.copy()
    y_train_Orignal=y_train.copy()
    x_cleanVal_Orignal=x_cleanVal.copy()
    y_cleanVal_Orignal=y_cleanVal.copy()

    feature_scope=[5]

    for iFeature in range(0, len(feature_scope)):
        featureNum = feature_scope[iFeature]
        iter=iter+1

        # feature selection
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
        AD.plot_coefficients_v2(clf, fea_name,resultPath,iter,str(name))
        AD.plot_abs_coefficients_v2(clf, fea_name,resultPath,iter,str(name))

        #correlation and covariance Matrix
        fea_name_str = listToStr = ' '.join([str(elem) for elem in fea_name])
        CF.corrmatrix_features_abs(df_train.loc[:,fea_name], 0, resultPath, iter,name)
        CF.covMatrix_features_abs(df_train.loc[:,fea_name], 0, resultPath, iter,name)

        #count number of burst features vs psd features
        [psd_num,burst_num]=AD.count_psd_vs_burst_number(fea_name)

        #print intersting values:
        print('current Subject: ' + str(name))
        print("Best: %f using %s" % (clf.best_score_, clf.best_params_))


        # '''Results''':
        #Train
        trainAccuracy = [str(round(clf.cv_results_['mean_train_accuracy_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_train_accuracy_score'][clf.best_index_], 2)) + ')']
        trainAUC = [str(round(clf.cv_results_['mean_train_roc_auc_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_train_roc_auc_score'][clf.best_index_], 2)) + ')']
        trainCohen = [str(round(clf.cv_results_['mean_train_cohen_kappa_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_train_cohen_kappa_score'][clf.best_index_], 2)) + ')']
        trainPrecision = [str(round(clf.cv_results_['mean_train_precision_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_train_precision_score'][clf.best_index_], 2)) + ')']
        trainRecall = [str(round(clf.cv_results_['mean_train_recall_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_train_recall_score'][clf.best_index_], 2)) + ')']
        trainSpecificity = [str(round(clf.cv_results_['mean_train_specificity'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_train_specificity'][clf.best_index_], 2)) + ')']

        #Test CV:
        testCVAccuracy = [str(round(clf.cv_results_['mean_test_accuracy_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_test_accuracy_score'][clf.best_index_], 2)) + ')']
        testCVAUC = [str(round(clf.cv_results_['mean_test_roc_auc_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_test_roc_auc_score'][clf.best_index_], 2)) + ')']
        testCohen = [str(round(clf.cv_results_['mean_test_cohen_kappa_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_test_cohen_kappa_score'][clf.best_index_], 2)) + ')']
        testPrecision = [str(round(clf.cv_results_['mean_test_precision_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_test_precision_score'][clf.best_index_], 2)) + ')']
        testRecall = [str(round(clf.cv_results_['mean_test_recall_score'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_test_recall_score'][clf.best_index_], 2)) + ')']
        testSpecificity = [str(round(clf.cv_results_['mean_test_specificity'][clf.best_index_], 2)) + '(+/-' + str(round(clf.cv_results_['std_test_specificity'][clf.best_index_], 2)) + ')']

        #Valditon Clean Data:
        y_cleanVal_pred = clf.predict(x_cleanVal)
        cleanValAccuracy =accuracy_score(y_cleanVal.ravel(), y_cleanVal_pred)
        cleanVal_roc_auc=roc_auc_score(y_cleanVal.ravel(),y_cleanVal_pred)
        cleanVal_Cohen=cohen_kappa_score(y_cleanVal.ravel(),y_cleanVal_pred)
        cleanVal_precision=precision_score(y_cleanVal.ravel(),y_cleanVal_pred)
        cleanVal_recall=recall_score(y_cleanVal.ravel(),y_cleanVal_pred)
        cleanVal_specificity=recall_score(y_cleanVal.ravel(),y_cleanVal_pred,pos_label=-1)

        line, df_results_new = AD.add2Excel_v4_gridSearch(0,psd_num,burst_num, trainAccuracy,trainAUC,trainPrecision,trainRecall,trainCohen,trainSpecificity,testCVAccuracy,testCVAUC,testPrecision,testRecall,testCohen,testSpecificity,cleanValAccuracy,cleanVal_roc_auc,cleanVal_precision,cleanVal_recall,cleanVal_Cohen,cleanVal_specificity,featureNum,clf.best_params_, line, df_results_new, name,math.floor(subject_number/6),'mean CV 5fold- ACC & AUC ')

        if name[2]=='3_2':
            arr_allelecAcc=df_results_new.testCV_accuracy[line-5:line+1]
            indHigestAcc=arr_allelecAcc.values.argmax()
            winnerElecInd=(line - 6 + indHigestAcc)
            df_results_new.loc[winnerElecInd, 'winnerElec'] = 1

if Group_flag:
    df_results_new.to_csv(path_or_buf=resultPath+'SVM__model_Group&allContacts_withOutOutliers_modelPerAllCV.csv')
else:
    df_results_new.to_csv(path_or_buf=resultPath+'SVM__model_perSubject&allContacts_withOutOutliers_modelPerAllCV.csv')

