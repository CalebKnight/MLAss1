

import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

from data import MakeDf

import matplotlib.pyplot as plt

from csv import writer

from features import Features


# We want to take


def MakeSVR(df, i):
    X = df.drop("Target Pressure (bar)", axis=1, inplace=False)
    y = df["Target Pressure (bar)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the SVR model
    # EPSILON IS HUGE

    c = 1
    if i == 2:
        c = 100
    if i == 1 or i == 4:
        c = 10
    g = 0.01
    if i == 3:
        g = 0.1

    svr = SVR(kernel='rbf', C=c, gamma=g, epsilon=0.01, cache_size=1000)
    svr.fit(X_train_scaled, y_train)

    return svr, scaler, X_test_scaled, y_test

def SVRGridSearch(df):
    X = df.drop("Target Pressure (bar)", axis=1)
    y = df["Target Pressure (bar)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
   

    # Initialize and train the SVR model
    # EPSILON IS HUGE

    svr = SVR(kernel='rbf', cache_size=1000)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'epsilon': [0.01, 0.1, 0.5, 1, 2, 5]
    }
    grid = GridSearchCV(svr, param_grid, verbose=0)
    grid.fit(X_train_scaled, y_train)
    print(grid.best_params_)
    print(grid.best_score_)


def GridSearchAllSensors():
    for i in range(1,6):
        df = MakeDf(i)
        SVRGridSearch(df)


def UseTrainedModelsOnUnknownTest(models):
    predictions = []

    for idx, model in enumerate(models):
        svr, scaler = model
        df = MakeDf(idx + 1, testing=True, fileName="test.csv")
        df =  df[Features]
        X_test = df
        X_test_scaled = scaler.transform(X_test)
        y_pred = svr.predict(X_test_scaled)
        # we need to combine the ids from X_test with the predictions
        ids = df.index
        for i in range(len(ids)):
            predictions.append([ids[i], y_pred[i]])
        
    return predictions

def MakeSVRModel(df, i, subModels):
    svr, scaler, X_test_scaled, y_test = MakeSVR(df, i)
    y_pred = svr.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Sensor {i} MSE: {mse}")
    print(f"Sensor {i} MAPE: {mape}")   
    subModels.append((svr, scaler))


def SVRPredict(models):
    predictions = UseTrainedModelsOnUnknownTest(models)
    predictions.sort(key=lambda x: x[0])
    submission = pd.DataFrame(predictions, columns=["ID","Target Pressure (bar)"])
    submission.to_csv("SVR.csv", index=False)
    return submission





    
    