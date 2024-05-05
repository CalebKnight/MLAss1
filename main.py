

from data import CheckData, CheckMultipleData, MakeDf, PlotMultipleSensorData, PlotSensorData, ProcessData, ReadFile 
from neural import  GridSearchAllNNModels, MakeNNModel, NNPredict
from svr import GridSearchAllSensors, MakeSVRModel, SVRGridSearch, SVRPredict
from gbdt import GBDTPredict, GridSearchAllSensorsGBDT, MakeGBDTModels
from features import Features

def SVR():
    models = []
    for i in range(1,6):
        df = MakeDf(i)
        features = Features.copy()
        features.append("Target Pressure (bar)")
        newDF = df[features]
        MakeSVRModel(newDF, i, models)

    SVRPredict(models)


def NN(repeatCount=1):
    averageMse = 0
    baseModels = []
    bestMape = 100
    for i in range(1,repeatCount + 1):
        models = []
        mse = []
        for j in range(1,6):
            df = MakeDf(j)
            MakeNNModel(df, j, models)
        for model in models:
            mse.append(model[2])
            mape = model[3]
        averageMse += sum(mse) / len(mse)
        if mape < bestMape:
            bestMape = mape
            baseModels = models
        print(f"Run {i} MAPE: {mape}")
    NNPredict(baseModels)
    
    averageMse /= repeatCount
    print(f"Average MSE: {averageMse}")
    print(f"Average RMSE: {averageMse ** 0.5}")
    print(f"Best MAPE: {bestMape}")
    
def GBDT():
    models = MakeGBDTModels(ProcessData())
    GBDTPredict(models)

def main():
    # CheckMultipleData(ProcessData())
    SVR()
    NN(3)
    GBDT()


    
    

if __name__ == "__main__":
    main()

# Remember to talk about splitting up data with SVM model and how there are too many params to split the data into submodels for a human to identify



# MEAN AVERAGE ERROR GOES UP WITH OVER TRAINING