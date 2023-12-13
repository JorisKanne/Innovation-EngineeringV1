# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:35:57 2023

@author: joris
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from plotly.offline import plot
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree


Data_input = pd.read_excel(r'C:\Users\joris\Toegepaste Wiskunde\Toegepaste wiskunde\Jaar 4\Module\DataV1.xlsx',)
Data_output = pd.read_excel(r'C:\Users\joris\Toegepaste Wiskunde\Toegepaste wiskunde\Jaar 4\Module\DataV1.xlsx',sheet_name='LungeTime')


y = Data_output['Time post'][Data_input['Reconstruction']=='Hamstring']
WorkingData = Data_input[Data_input['Reconstruction']=='Hamstring']

WorkingData = WorkingData.iloc[: , :-1]

WorkingData = pd.get_dummies(WorkingData, columns = ['Sex'])
corr_matrix = WorkingData.corr()
fig1 = px.imshow(corr_matrix,text_auto=True)
plot(fig1)

WorkingData = WorkingData.drop(['ID','WeightPost'],axis=1)

corr_matrix = WorkingData.corr()
corr_matrix = round(corr_matrix,1)
fig2 = px.imshow(corr_matrix,text_auto=True)
fig2.update_layout(coloraxis_colorbar_x=0.75)
plot(fig2)

#%%
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.cluster import DBSCAN

#WorkingData[['Age','Height','Weight']] = np.log(WorkingData[['Age','Height','Weight']])
WorkingData['Sex_F'] = WorkingData['Sex_F'].astype(int)
WorkingData['Sex_M'] = WorkingData['Sex_M'].astype(int)
X = WorkingData

pca = PCA(n_components = 3)
X_PC = pca.fit_transform(X)
df = pd.DataFrame(X_PC)
# Filling in the parameter n(minpts)
n = 6
# Calculating epsilon based on the given paramater n by using the elbow method.
nearest_neighbors = NearestNeighbors(n_neighbors = n)
neighbours = nearest_neighbors.fit(df)
distances, IndexBuren = neighbours.kneighbors(df)
distances = np.sort(distances[:, n - 1], axis = 0)
kl = KneeLocator(np.arange(len(distances)), distances, curve="convex")
eps = float(kl.all_elbows_y[0])

# Actually doing the clustering, by using a function in the Sklearn toolbox
clustering = DBSCAN(eps=eps, min_samples=n).fit(df)
Cluster = clustering.labels_
fig = px.scatter_3d(x=X_PC[:,0],y=X_PC[:,1],z=X_PC[:,2], opacity=0.8,color=Cluster)
fig.update_layout(
    title={
    'text': "Clustering",
    'y':0.9,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},
    scene = dict(
                    xaxis_title='X_PC',
                    yaxis_title='Y_PC',
                    zaxis_title='Z_PC'),
    font=dict(
        family="Arial",
        size=14,
        color="black"
    )
)
plot(fig)
#%%

fig = px.scatter_3d(x=X_PC[:,0],y=X_PC[:,1],z=X_PC[:,2], opacity=0.8,color=y,range_color=(min(y),max(y)),
                        colorbar=dict(
        title="Surface Heat"
    ))
fig.update_layout(
    title={
    'text': "Dimension reduction of the 5 variables",
    'y':0.9,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},
    scene = dict(
                    xaxis_title='X_PC',
                    yaxis_title='Y_PC',
                    zaxis_title='Z_PC'),
    font=dict(
        family="Arial",
        size=14,
        color="black"
    )
)
fig.update_layout(coloraxis_colorbar_x=0.75)

plot(fig)
#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1)


#%%
reg = LinearRegression().fit(X_train, y_train)

ypred = reg.predict(X_test)
print(r2_score(y_test, ypred))
#%%
method = 'gini'
random_state = 2
max_depth = 3

DTR = tree.DecisionTreeRegressor(max_depth = max_depth, random_state = random_state)
DTR = DTR.fit(X_train, y_train)
y_pred = DTR.predict(X_test)

print(r2_score(y_test, y_pred))

#%%

def scoring_function(model, X, y):
    k = model.get_params()['max_depth']
    n = len(y)
    y_pred = model.predict(X)
    y_true = y
    score_ = 1 - (n - 1)/(n - k - 1) * (1 - r2_score(y_true, y_pred))
    return score_

from sklearn.model_selection import GridSearchCV
rfr = RandomForestRegressor(random_state = 1)
Xlist = np.linspace(1,200,10,dtype=int)
Ylist = np.linspace(1,4,4,dtype=int)

hyper_parameters = { 
    'n_estimators': Xlist, ## n_estimators is equal to the number of trees used 
    'max_depth' : Ylist
}

GS_rfr = GridSearchCV(estimator = rfr, param_grid = hyper_parameters,cv=2, scoring = scoring_function)
GS_rfr.fit(X_train, y_train)
ypred = GS_rfr.predict(X_test)
print(r2_score(y_test, ypred))

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


model = Sequential()
model.add(Dense(10, input_dim=5, activation='sigmoid'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='SGD', loss='mean_squared_error')

X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')

history = model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)

print(model.summary())
predictions = model.predict(X_test)
print(r2_score(y_test, predictions))
print(mean_squared_error(y_test,predictions))
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


plt.plot(hist['epoch'],hist['loss'])
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.title('Loss function over epochs')
plt.show()



