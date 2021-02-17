#Daniel Sand

import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np


def corrmatrix_features_abs(df, metaDataCol_N,resultPath, iter,name):
    plt.close('all')
    df_short = df.iloc[:, metaDataCol_N:].copy()

    correlation_mat_abs = abs(df_short.corr())

    sum_corr=np.tril(correlation_mat_abs, -1).sum()
    mean_corr=sum_corr/10
    print(mean_corr)

    plt.figure(figsize=(25, 20))
    ax = sns.heatmap(correlation_mat_abs, vmin=0, vmax=1, annot=True, linewidths=.5)
    ax.tick_params(labelsize=10)
    plt.title('meanCorrValue_withoutDiagonal'+str(mean_corr))



    path=resultPath+'correlation_matrix_abs/'
    if not os.path.exists(path):
        os.makedirs(path)
    strName = (', '.join(str(x) for x in [name]))
    fileName_corr= path + strName + '_pearsonCorr_'+'iter_' + str(iter) + '.png'
    plt.savefig(fileName_corr, dpi=300)


    return


def covMatrix_features_abs(df, metaDataCol_N, resultPath, iter, name):
    plt.close('all')
    df_short = df.iloc[:, metaDataCol_N:].copy()

    cov_mat_abs = abs(df_short.cov())

    plt.figure(figsize=(25, 20))
    ax = sns.heatmap(cov_mat_abs, annot=True, linewidths=.5)
    ax.tick_params(labelsize=10)

    # save fig

    path = resultPath + 'cov_matrix/'
    if not os.path.exists(path):
        os.makedirs(path)
    strName = (', '.join(str(x) for x in [name]))
    fileName = path + strName + 'iter_' + str(iter) + '.png'

    plt.savefig(fileName, dpi=300)
    return