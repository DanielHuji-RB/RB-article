#Daniel Sand



import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
import pandas as pd




def ROC_LOO_binary(ylabels, scores):


    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import confusion_matrix

    pos_label_array=[0,1]
    for i in  range(0,2):
        pos_label = pos_label_array[i]

        #intliazion
        results = type('', (), {})()
        results.train_roc_auc=np.array([])
        results.train_precision = np.array([])
        results.train_recall= np.array([])
        results.train_accuracy= np.array([])
        results.train_specificity= np.array([])
        results.train_senstivity= np.array([])
        results.test_TrueLabel=np.array([])
        results.test_Pred=np.array([])

        index=-1
        loo = LeaveOneOut()
        loo.get_n_splits(scores)

        # loop on LOO
        for train_index, test_index in loo.split(scores):
            index=index+1
            X_train, X_test = scores[train_index], scores[test_index]
            y_train, y_test = ylabels[train_index], ylabels[test_index]

            fpr, tpr, thresholds = metrics.roc_curve(y_train, X_train,pos_label=pos_label)
            results.train_roc_auc= np.append(results.train_roc_auc, auc(fpr, tpr))

            optimal_idx = np.argmax(tpr + (1-fpr))
            optimal_threshold = thresholds[optimal_idx]


            indxABoveTH=X_train>=optimal_threshold
            trainPred=np.empty(indxABoveTH.shape)
            trainPred[:]=np.nan
            trainPred[indxABoveTH==True]=pos_label
            trainPred[indxABoveTH==False]=1 - pos_label
            train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train, trainPred).ravel()

            results.train_senstivity=np.append(results.train_senstivity,tpr[optimal_idx])
            results.train_specificity=np.append(results.train_specificity,1-fpr[optimal_idx])
            results.train_precision = np.append(results.train_precision,train_tp / (train_tp + train_fp))
            results.train_recall = np.append(results.train_recall,(train_tp / (train_tp + train_fn)))
            results.train_accuracy = np.append(results.train_accuracy,(train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn))

            test_PosOrNeg         = pos_label if X_test >= optimal_threshold else (1 - pos_label)
            results.test_TrueLabel =np.append(results.test_TrueLabel,y_test)
            results.test_Pred=np.append(results.test_Pred,test_PosOrNeg)


        #Test Mesurment caclulation
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(results.test_TrueLabel, results.test_Pred).ravel()
        results.test_precision = test_tp / (test_tp + test_fp)
        results.test_recall = test_tp / (test_tp + test_fn)
        results.test_accuracy=(test_tp+test_tn)/(test_tp+test_tn+test_fp+test_fn)


        print("AUC mean:"+str(results.train_roc_auc.mean()))
        if sum(results.train_roc_auc >= 0.5) >= results.train_roc_auc.size/2:
            print('noNeed to reRun the function with different pos_label becouse most of the itearation was in the right classfication')
            break # this break is if the direction of the postive label was ok ( auc >0.5)
        else:
            print(' Warning: There are more iteartions with train auc under 0.5, the function will replace change pos_label from 0 to 1  ')
            print('num of oppsite direction (auc less then 0.5):'+str(sum(results.train_roc_auc < 0.5)))



    return results





def ROC_CV_binary(ylabels, scores,ylabels_cleanVal, scores_cleanVal):


    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold

    cleanVal_flag=1

    ylabels[ylabels==-1]=0
    if cleanVal_flag:
        ylabels_cleanVal[ylabels_cleanVal == -1] = 0


    pos_label_array=[0,1]
    for i in  range(0,2):
        pos_label = pos_label_array[i]

        #intliazion
        results = type('', (), {})()
        results.optimal_threshold=np.array([])
        results.train_roc_auc=np.array([])
        results.train_precision = np.array([])
        results.train_recall= np.array([])
        results.train_accuracy= np.array([])
        results.train_specificity= np.array([])
        results.train_senstivity= np.array([])
        results.test_precision=np.array([])
        results.test_recall=np.array([])
        results.test_accuracy=np.array([])
        results.test_auc=np.array([])



        index=-1
        skf = StratifiedKFold(n_splits=5,random_state=None)
        skf.get_n_splits(scores, ylabels)


        for train_index, test_index in skf.split(scores, ylabels):

            index=index+1
            X_train, X_test = scores[train_index], scores[test_index]
            y_train, y_test = ylabels[train_index], ylabels[test_index]

            fpr, tpr, thresholds = metrics.roc_curve(y_train, X_train,pos_label=pos_label)
            results.train_roc_auc= np.append(results.train_roc_auc, auc(fpr, tpr))#adding auc for this train iteration


            optimal_idx = np.argmax(tpr + (1-fpr))
            optimal_threshold = thresholds[optimal_idx]
            results.optimal_threshold=np.append(results.optimal_threshold,optimal_threshold)


            #Train mesurment calculation
            indxABoveTH=X_train>=optimal_threshold
            trainPred=np.empty(indxABoveTH.shape)
            trainPred[:]=np.nan
            trainPred[indxABoveTH==True]=pos_label
            trainPred[indxABoveTH==False]=1 - pos_label
            train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train, trainPred).ravel()

            results.train_senstivity=np.append(results.train_senstivity,tpr[optimal_idx])
            results.train_specificity=np.append(results.train_specificity,1-fpr[optimal_idx])
            results.train_precision = np.append(results.train_precision,train_tp / (train_tp + train_fp))
            results.train_recall = np.append(results.train_recall,(train_tp / (train_tp + train_fn)))
            results.train_accuracy = np.append(results.train_accuracy,(train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn))

            #Test- collect the value of the test refer to the train TH
            test_PosOrNeg = np.zeros((X_test.size, 1))
            test_PosOrNeg[:] = np.nan

            test_PosOrNeg_temp = np.array(X_test >= optimal_threshold )
            test_PosOrNeg[test_PosOrNeg_temp==True]=pos_label
            test_PosOrNeg[test_PosOrNeg_temp==False]=1 - pos_label



            #Test Mesurment caclulation
            results.test_precision=np.append(results.test_precision,precision_score(y_test,test_PosOrNeg))
            results.test_recall=np.append(results.test_recall,recall_score(y_test,test_PosOrNeg))
            results.test_accuracy=np.append(results.test_accuracy,accuracy_score(y_test,test_PosOrNeg))
            results.test_auc=np.append(results.test_auc,roc_auc_score(y_test,test_PosOrNeg))


        print("AUC mean:"+str(results.train_roc_auc.mean()))
        if sum(results.train_roc_auc >= 0.5) >= results.train_roc_auc.size/2:
            print('noNeed to reRun the function with different pos_label becouse most of the itearation was in the right classfication')
            break #
        else:
            print(' Warning: There are more iteartions with train auc under 0.5, the function will replace change pos_label from 0 to 1  ')
            print('num of oppsite direction (auc less then 0.5):'+str(sum(results.train_roc_auc < 0.5)))


    '''clean Valdtion Mesurment caclulation'''
    if cleanVal_flag:

        # inlitize
        cleanVal_PosOrNeg = np.zeros((scores_cleanVal.size, 1))
        cleanVal_PosOrNeg[:] = np.nan

        # finding mean TH
        cleanVal_PosOrNeg_temp = np.array(
            scores_cleanVal >= results.optimal_threshold.mean())  #
        cleanVal_PosOrNeg[cleanVal_PosOrNeg_temp == True] = pos_label
        cleanVal_PosOrNeg[cleanVal_PosOrNeg_temp == False] = 1 - pos_label

        # calculting measrments
        results.cleanVal_precision = precision_score(ylabels_cleanVal, cleanVal_PosOrNeg)
        results.cleanVal_recall = recall_score(ylabels_cleanVal, cleanVal_PosOrNeg)
        results.cleanVal_accuracy = accuracy_score(ylabels_cleanVal, cleanVal_PosOrNeg)

    return results


