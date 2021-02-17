#daniel sand


import numpy as np
import pandas as pd
import os
# import matlab.engine
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
from Functions import arrangeData as AD
from Functions import FeaturesSelection_script as FSS
from sklearn.model_selection import StratifiedKFold
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

from Functions import arrangeData as AD
from Functions import mrmr as M


def DL_v1(x_train,y_train,df_xTrain, x_val,y_val, x_test,y_test, feature_scope, df_results_new, line, name, iter):
    strName_forWeigths = (', '.join(str(x) for x in [name]))

    #inltizion
    x_train_orginal = x_train.copy()
    y_train_orginal= y_train.copy()
    x_val_orginal = x_val.copy()
    y_val_orginal = y_val.copy()
    x_test_orginal = x_test.copy()
    y_test_orginal = y_test.copy()
    modelCounter=0
    # eng = matlab.engine.start_matlab()
    valAcc=0

    for iFeature in range(0, len(feature_scope)):
           #Feature Selection
            featureNum = feature_scope[iFeature]
            FSmodel = M.mRMR(nFeats=featureNum)
            modelFit = FSmodel.fit(x_train_orginal, y_train_orginal)
            x_train=modelFit._transform(x_train_orginal)
            x_val = modelFit._transform(x_val_orginal)
            x_test = modelFit._transform(x_test_orginal)


           #not sure this is important
            y_train= y_train_orginal.copy()
            y_val = y_val_orginal.copy()
            y_test = y_test_orginal.copy()

            '#_model paramters'

            loss_scope=['mean_squared_error','hinge','categorical_crossentropy']
            activation_scope=['tanh','relu','sigmoid']

            for iLoss in range(0,len(loss_scope)):
                lossName=loss_scope[iLoss]
                for iActivation in range(0,len(activation_scope)):
                    activationName=activation_scope[iActivation]
                    # optimzer_scope=[ 'Adam', 'Adamax','SGD','Adagrad' ,'Adadelta','RMSprop' ,'Nadam']# all optimzers beside 'TFOptimizer',
                    optimzer_scope=[ 'Adam','Adamax','SGD']# all optimzers beside 'TFOptimizer',
                    for iOptimzer in range(0, len(optimzer_scope)):
                        optimzerName=optimzer_scope[iOptimzer]
                        dropoutV_scope=[0.2,0.4,0.6,0.8]
                        for iDropOut in range (0, len(dropoutV_scope)):
                            dropoutV=dropoutV_scope[iDropOut]

                            batch_size_n_scope = [32]
                            for iBatch in range(0, len(batch_size_n_scope)):
                                batch_size_n = batch_size_n_scope[iBatch]
                                modelCounter=modelCounter+1

                                epochs=500
                                '''Train model:'''
                                N_nodes=50
                                # first sub_nn
                                input_data_1 = Input(shape=(featureNum,))
                                x1 = Dense(N_nodes, activation=activationName)(input_data_1)  # relu  sigmoid
                                for iInputData in range(0, 4):
                                    x1 = Dropout(dropoutV)(x1)
                                    x1 = Dense(50, activation=activationName)(x1)
                                x1 = Dropout(dropoutV)(x1)
                                top_nn_output = Dense(15, activation=activationName)(x1)
                                output_layer = Dense(1)(top_nn_output)

                                model = Model(input=[input_data_1], outputs=[output_layer])
                                model.compile(optimizer=optimzerName, loss=lossName, metrics=['accuracy'])# loss='binary_crossentropy'


                                # checkpoint
                                filepath = "weights.best"+strName_forWeigths+str(iter)+".hdf5"
                                checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,  mode='max')
                                # early stop
                                earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=100, verbose=1, mode='auto')
                                callbacks_list = [earlystop,checkpoint]
                                history = model.fit([x_train],[y_train.ravel()], epochs=epochs, batch_size=batch_size_n ,callbacks=callbacks_list, validation_data=([x_val], [y_val.ravel()]))#need to drop off the second label
                                valScore_temp, valAcc_temp = model.evaluate([x_val], [y_val.ravel()], batch_size=batch_size_n).copy()

                                if  np.array(model.history.history['val_acc']).argmax() >20:
                                     model.load_weights(filepath)

                                valScore_temp, valAcc_temp = model.evaluate([x_val], [y_val.ravel()], batch_size=batch_size_n).copy()
                                trainScore_temp, trainAcc_temp = model.evaluate([x_train], [y_train.ravel()], batch_size=batch_size_n).copy()

                                if (valAcc_temp > valAcc) and (trainAcc_temp>0.6):

                                    modelH_Params = type('', (), {})()

                                    valAcc = valAcc_temp
                                    modelH_Params.Val_score=valScore_temp.copy()
                                    modelH_Params.Val_accuracy=valAcc_temp.copy()

                                    modelH_Params.Train_score, modelH_Params.Train_accuracy = model.evaluate([x_train], [y_train.ravel()],batch_size=batch_size_n).copy()
                                    modelH_Params.Test_score, modelH_Params.Test_accuracy = model.evaluate([x_test], [y_test.ravel()],batch_size=batch_size_n).copy()
                                    print('Test_acc :' + str(modelH_Params.Test_accuracy))


                                    #hyperParams
                                    modelH_Params.Model=model
                                    modelH_Params.Modelhistory=history
                                    modelH_Params.Modelname='Complex_NN_Model'
                                    modelH_Params.featureSelection_N= featureNum
                                    modelH_Params.optimzerName = optimzerName
                                    modelH_Params.epochs = epochs
                                    modelH_Params.Reg = 'nan'#reg
                                    modelH_Params.batch_size_n = batch_size_n
                                    modelH_Params.activationName = activationName
                                    modelH_Params.lossfunction=lossName
                                    modelH_Params.dropoutV=dropoutV


    line, df_results_new = AD.add2Excel_v2_DL(modelH_Params, line,  df_results_new, name, str(iter))

    return modelH_Params,df_results_new,line