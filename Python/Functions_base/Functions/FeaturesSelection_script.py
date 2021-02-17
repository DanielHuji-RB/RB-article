#import


#MRMR function

def FS_MRMR(x_train,y_train,featureNum,df_xTrain,eng):

    import numpy as np
    import matlab.engine

    # eng = matlab.engine.start_matlab()
    label_matlab = matlab.double(y_train.tolist())
    train_matlab = matlab.double(x_train.tolist())

    #MRMR function
    fea = eng.mrmr_miq_d(train_matlab, label_matlab, featureNum)

    fea = np.array(fea._data.tolist()).astype(int)
    fea = fea - 1  # dealing with the differenc between matlab index start at 1 and python at 0
    print(fea)


    if ((np.size(x_train, 1) != len(df_xTrain.columns))):
        print('ERROR: df_xTrain must be the same column number as x_train and to include the coulmns name')
        quit()

    fea_name = list(df_xTrain.iloc[:, fea])
    print(fea_name)

    return fea, fea_name
