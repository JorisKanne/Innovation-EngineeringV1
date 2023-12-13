# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:56:38 2023

@author: joris
"""

# Use scikit-learn to grid search the number of neurons
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from scikeras.wrappers import KerasRegressor
import pandas as pd
from tensorflow.keras.constraints import MaxNorm

# Function to create model, required for KerasClassifier
def create_model(neurons1,neurons2,activation1 = 'relu',activation2 = 'relu'):
    # create model
    model = Sequential()
    model.add(Dense(neurons1, input_shape=(5,), kernel_initializer='uniform', activation=activation1))
    model.add(Dense(neurons2, activation=activation2))
    model.add(Dense(1, activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])
    return model
# Compile model
# fix random seed for reproducibility
seed = 7
tf.random.set_seed(seed)
# load dataset
Data_input = pd.read_excel(r'C:\Users\joris\Toegepaste Wiskunde\Toegepaste wiskunde\Jaar 4\Module\DataV1.xlsx',)
Data_output = pd.read_excel(r'C:\Users\joris\Toegepaste Wiskunde\Toegepaste wiskunde\Jaar 4\Module\DataV1.xlsx',sheet_name='LungeTime')

y = Data_output['Time post'][Data_input['Reconstruction']=='Hamstring']
WorkingData = Data_input[Data_input['Reconstruction']=='Hamstring']

WorkingData = WorkingData.iloc[: , :-1]
WorkingData = WorkingData.drop(['ID','WeightPost'],axis=1)
WorkingData = pd.get_dummies(WorkingData, columns = ['Sex'])
WorkingData['Sex_F'] = WorkingData['Sex_F'].astype(int)
WorkingData['Sex_M'] = WorkingData['Sex_M'].astype(int)
X = WorkingData
Y = y
# create model
model = KerasRegressor(model=create_model, epochs=100, batch_size=10, verbose=0)
# define the grid search parameters
neurons = [i+1 for i in range(5)]
activation = ['relu', 'tanh', 'sigmoid','hard_sigmoid']
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(model__neurons1=neurons,model__neurons2 = neurons,model__activation1=activation,model__activation2=activation,optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))