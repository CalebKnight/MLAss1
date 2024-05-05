from math import log2
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from data import MakeDf



from features import Features

def MakeGBDTModels(dfArray):
    models = []
    for i in range (1, 6):
        newDf = dfArray[i-1].copy()
        features = Features.copy()
        features.append("Target Pressure (bar)")
        newDf = newDf[features]
        model, scaler, X_test, y_test = MakeGBDT(newDf)
        y_pred = model.predict(scaler.transform(X_test))
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        models.append([model, scaler, mse, mape])
        print(f"Sensor {i} MSE: {mse}")
        print(f"Sensor {i} MAPE: {mape}")
    return models

def UseTrainedModelsOnUnknownTest(models):
    predictions = []

    for idx, model in enumerate(models):
        model, scaler, mse, mape = model
        df = MakeDf(idx + 1, testing=True, fileName="test.csv")
        X_test =  df[Features.copy()]
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        # we need to combine the ids from X_test with the predictions
        ids = df.index
        for i in range(len(ids)):
            predictions.append([ids[i], y_pred[i]]) 
    return predictions

def GBDTPredict(models):
    predictions = UseTrainedModelsOnUnknownTest(models)
    predictions.sort(key=lambda x: x[0])
    submission = pd.DataFrame(predictions, columns=["ID","Target Pressure (bar)"])
    submission.to_csv("submissionGBDT.csv", index=False)
    return submission

   

def MakeGBDT(df):
    scaler = StandardScaler()
    X = df.drop(columns=["Target Pressure (bar)"])
    y = df["Target Pressure (bar)"]
    features_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(features_scaled, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    depth = int(log2(len(df.columns)))
    model = GradientBoostingRegressor(loss="squared_error", n_estimators=600, max_depth=depth, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model, scaler, X_test, y_test

def GBDTGridSearch(df):
    scaler = StandardScaler()
    X = df.drop(columns=["Target Pressure (bar)"])
    y = df["Target Pressure (bar)"]
    features_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(features_scaled, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    depth = (len(df.columns)) ** 2
    model = GradientBoostingRegressor(loss="squared_error")
    param_grid = {
        'n_estimators': [100, 300, 600],
        'learning_rate': [0.001, 0.01, 0.1]
    }
    grid = GridSearchCV(model, param_grid, verbose=1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_ )

def GridSearchAllSensorsGBDT():
    for i in range(1,6):
        df = MakeDf(i)
        GBDTGridSearch(df)


