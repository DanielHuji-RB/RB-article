

def dataEvaluation_func(Xscores,Ylabels):

    from sklearn.model_selection import LeaveOneOut,cross_validate
    from sklearn import model_selection,svm
    from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, auc, \
        make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix,precision_macro,recall_macro



    loocv = model_selection.LeaveOneOut()

    results = type('', (), {})()

    scorsList = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }

    model = svm.SVC(kernel='linear', C=1, random_state=0)
    results = cross_validate(model, Xscores, Ylabels, scoring=scorsList, cv=loocv, return_train_score=True)
    return results,scorsList


