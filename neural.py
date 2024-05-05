import pandas as pd
import tensorflow as tf
import keras
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor

from data import MakeDf


from features import Features

def MakeNN(df, i):
    scaler = StandardScaler()
    newDF = df[Features].copy()
    X = newDF
    y = df["Target Pressure (bar)"]
    features_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(features_scaled, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    l1Inputs = keras.Input(shape=(int(X.shape[1]),))
    maxNorm = keras.constraints.MaxNorm(0.05)
    constraint = maxNorm

    l1 = keras.layers.Dense(int(X.shape[1]), activation='relu', kernel_initializer="he_normal", kernel_constraint=constraint)(l1Inputs)
    l2 = keras.layers.Dense(int(X.shape[1]), activation='relu', kernel_initializer="he_normal", kernel_constraint=constraint)(l1)
    outputs = keras.layers.Dense(1, activation="linear")(l2)
    model = keras.Model(inputs=l1Inputs, outputs=outputs)

    # We use RMSprop because it handles noisy data effectively

    loss = keras.losses.MeanAbsolutePercentageError()
    model.compile(optimizer='rmsprop', loss=loss, metrics=[loss])
    model = KerasRegressor(model, batch_size=16, epochs=25)
    model.fit(X_train, y_train, epochs=25, verbose=0)
    return model , scaler, X_test, y_test

def MakeNNModel(df, i, models):
    model, scaler, X_test, y_test = MakeNN(df, i)
    y_pred = model.predict(scaler.transform(X_test))
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    models.append([model, scaler, mse, mape])

def UseTrainedNNModelsOnUnknownTest(models):
    predictions = []

    for idx, model in enumerate(models):
        model, scaler, mse, mape = model
        df = MakeDf(idx + 1, testing=True, fileName="test.csv")
        df = df[Features].copy()
        X_test = df
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        # we need to combine the ids from X_test with the predictions
        ids = df.index
        for i in range(len(ids)):
            predictions.append([ids[i], y_pred[i]]) 
    return predictions


def NNPredict(models):
    predictions = UseTrainedNNModelsOnUnknownTest(models)
    predictions.sort(key=lambda x: x[0])
    submission = pd.DataFrame(predictions, columns=["ID","Target Pressure (bar)"])
    submission.to_csv("submissionNN1.csv", index=False)
    return submission


def GridSearchNNModel(df):
    scaler = StandardScaler()
    X = df.drop(columns=["Target Pressure (bar)"], inplace=False)
    y = df["Target Pressure (bar)"]
    features_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(features_scaled, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    l1Inputs = keras.Input(shape=(int(X.shape[1]),))


    l1 = keras.layers.Dense(int(X.shape[1]), activation='relu')(l1Inputs)
    l2 = keras.layers.Dense(int(X.shape[1]), activation='relu')(l1)
    outputs = keras.layers.Dense(1, activation="linear")(l2)
    model = keras.Model(inputs=l1Inputs, outputs=outputs)

    # We use RMSprop because it handles noisy data effectively

    
    loss = keras.losses.MeanAbsolutePercentageError()

    model.compile(optimizer='adam', loss=loss, metrics=[loss])
    model = KerasRegressor(model)

    param_grid = {
        "epochs": [10, 50, 100, 250],
        "batch_size": [16],
        "optimizer": ["rmsprop"],
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_)


    
def GridSearchAllNNModels():
    for i in range(1,6):
        df = MakeDf(i)
        GridSearchNNModel(df)

    
    



