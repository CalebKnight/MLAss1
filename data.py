import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def ReadFile(fileName = "train.csv"):
    df = pd.read_csv(fileName)
    return df


def RemoveNullValues(df):
    # Simply remove NA columns given they have too much data missing to be useful
    df.dropna(inplace=True)
    return df


def CheckDuplicates(df):
    # Only drop duplicates for columns other than sensor data
    newDF = df.drop(["Sensor ID", "Sensor Position Side", "Sensor Position x", "Sensor Position y", "Sensor Position z"], axis=1)
    newDF = newDF.drop_duplicates()
    # Merge results back into original dataframe
    newDF.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df



def RemoveExcessivelyHighTargetPressure(df):
    # Values over one bar are not valid but including them means more data for the model
    # Utilising a sigmoid function here infact makes the model worse
    df = df[df["Target Pressure (bar)"] <= 1.1]
    df = df[df["Target Pressure (bar)"] > 0]
    return df

# Remove tanks that are excessively large to help with vapour based features (vapour amount is linear to tank volume)
# Lets instead move their decimal place 2 back e.g 2042 becomes 20.42
def RemoveExcessivelyLargeTanks(df):
    df = df[df["Tank Volume"] < 100]
    return df


def RemoveExcessivelyHighTankPressure(df):
    highDf = df[df["Tank Failure Pressure (bar)"] > 100].copy()
    highDf= highDf["Tank Failure Pressure (bar)"] / 100
    df = df[df["Tank Failure Pressure (bar)"] < 100]
    df = pd.concat([df, highDf])
    df = df[df["Tank Failure Pressure (bar)"] > 0]
    return df

# see count of values 0-0.1 0.1-0.2 ....
def PlotTargetPressure(df):
    df["Target Pressure (bar)"].hist(bins=100)
    plt.xlabel("Target Pressure (bar)")
    plt.ylabel("Count")
    plt.show()



def FixMissSpelledSuperheatedOrSubcooled(df):
    # If the string is similar to Superheated, we will assume it is superheated
    # If the string is similar to Subcooled, we will assume it is subcooled
    newDf = df.copy()
    newDf["Status"] = df["Status"].apply(lambda x: "Superheated" if "Super" in x else "Subcooled")
    newDf["Status"] = newDf["Status"].apply(lambda x: 1 if "Superheated" in x else 0)
    return newDf


# Engineer features that are more useful than the original features
def EngineerFeatures(df, testing=False):
    newDf = df.copy()
    # Volume of Tank
    newDf["Tank Volume"] = df["Tank Width (m)"] * df["Tank Height (m)"] * df["Tank Length (m)"]
    # Surface Area of Obstacle
    newDf["Obstacle SA"] = df["Obstacle Width (m)"] * df["Obstacle Height (m)"]
    # Obstacle Distance to Surface Area (Measures the cone of the obstacle in relation to distance to BLEVE)
    newDf["Obstacle Distance to Surface Area"] = df["Obstacle Distance to BLEVE (m)"] * newDf["Obstacle SA"]
    # A measure of how much pressure is in the tank
    newDf["TFP/TV"] = df["Tank Failure Pressure (bar)"] / newDf["Tank Volume"]
    # How much vapour is in the tank
    newDf["Vapour Amount"] = df["Vapour Height (m)"] / df["Tank Height (m)"]
    # Identifying how high of an angle the sensor is at compared to the y position
    newDf["Y*Angle"] = df["Obstacle Angle"] * df["Sensor Position y"]
    
    # Inferring the relationship between the distance to the BLEVE, the height of the BLEVE and the angle of the obstacle
    newDf["DB*BH*A"] = df["Obstacle Distance to BLEVE (m)"] * df["BLEVE Height (m)"] * df["Obstacle Angle"]

    if not testing:
        newDf = newDf[newDf["Vapour Amount"] < 1]
    newDf["Vapour Volume"] = newDf["Vapour Amount"] * newDf["Tank Volume"]
    newDf.drop("Vapour Amount", axis=1, inplace=True)
    return newDf

# Select the sensor data for a given sensor number
def ChooseSensors(df, sensorNumber):
    df = df[df["Sensor Position Side"] == sensorNumber]
    return df


# We want to shrink our range of sensors to remove outliers
def RemoveFaultySensors(df, sensorNumber):
    # This sensor is on the left side and thus can have -y values
    if sensorNumber == 3:
        df = df[df["Sensor Position y"] <= 5]
    # This sensor is on the right side and thus can have high +y values but not low positive values
    elif sensorNumber == 5:
        df = df[df["Sensor Position y"] >= 10]
    # The sensors are in the middle and thus can have a range of values between 0 and 10
    else:
        df = df[df["Sensor Position y"] > 0]
        df = df[df["Sensor Position y"] < 10]
    return df
        



def DropUselessColumns(df):
    # Get rid of features now being used for volume (A more useful feature than including all 3 dimensions of the tank and obstacle)
    df.drop(["Tank Width (m)", "Tank Height (m)", "Tank Length (m)"], axis=1, inplace=True)
    df.drop(["Obstacle Width (m)", "Obstacle Height (m)", "Obstacle Thickness (m)"], axis=1, inplace=True)
    
    # Get rid of features that are not useful for our models
    df.drop(["Liquid Critical Pressure (bar)", "Liquid Critical Temperature (K)", "Liquid Boiling Temperature (K)", "Obstacle Angle", "Liquid Ratio (%)", "Vapour Temperature (K)", "BLEVE Height (m)"], axis=1, inplace=True)
    return df


# Plot a 3D graph of the sensor data
def PlotSensorData(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df["Sensor Position x"], df["Sensor Position y"], df["Sensor Position z"])
    plt.xlabel("Sensor Position x")
    plt.ylabel("Sensor Position y")
    plt.show()

def CheckData(df, name):

    cols = df.columns

    cols = np.flip(cols)

    for col in cols:
        X,y = df[col], df["Target Pressure (bar)"]
        plt.scatter(X, y, s=0.5, alpha=0.5)
        # plt.bar(X, y)
        plt.xlabel(col)
        plt.ylabel("Target Pressure (bar)")
        plt.title(name)
        plt.show()



def PlotMultipleSensorData(dfArray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for df in dfArray:
        ax.scatter(df["Sensor Position x"], df["Sensor Position y"], df["Sensor Position z"])
    plt.legend(["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4", "Sensor 5"])
    plt.xlabel("Sensor Position x")
    plt.ylabel("Sensor Position y")
    plt.show()




def CheckMultipleData(dfArray):
    cols = dfArray[0].columns
    # reverse order of df columns
    cols = np.flip(cols)
    for col in cols:
        # we want to plot each arrays values as a different colour so we can loop through once with all 5 sensors
        plt.scatter(dfArray[0][col], dfArray[0]["Target Pressure (bar)"], s=0.5, alpha=0.5, c="red")
        plt.scatter(dfArray[1][col], dfArray[1]["Target Pressure (bar)"], s=0.5, alpha=0.5, c="blue")
        plt.scatter(dfArray[2][col], dfArray[2]["Target Pressure (bar)"], s=0.5, alpha=0.5, c="green")
        plt.scatter(dfArray[3][col], dfArray[3]["Target Pressure (bar)"], s=0.5, alpha=0.5, c="yellow")
        plt.scatter(dfArray[4][col], dfArray[4]["Target Pressure (bar)"], s=0.5, alpha=0.5, c="purple")
        plt.legend(["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4", "Sensor 5"])
        plt.xlabel(col)
        plt.ylabel("Target Pressure (bar)")
        plt.title("All Sensors")
        plt.show()




def ProcessData(targetSensor=None, Testing=False):
    i = 1
    # For testing purposes we can specify a target sensor
    if(targetSensor):
        i = targetSensor
    dfs = []
    # Var for counting samples when testing training data size
    sampleCount = 0
    for sensorNumber in range(i, 6):
        df = MakeDf(sensorNumber, testing=Testing, fileName="train.csv")
        dfs.append(df)
        sampleCount += len(df)
    return dfs

        

# vapour height, obstacle thickness, distance to obstacle
def MakeDf(sensorNumber, testing=False, fileName = "train.csv"):
    df = ReadFile(fileName)
    if not testing:
        df = RemoveNullValues(df)
    df = FixMissSpelledSuperheatedOrSubcooled(df)
    df = EngineerFeatures(df, testing)
    # We do not want to alter the testing data with modifications we make to the training data
    if not testing:
        df = RemoveExcessivelyHighTargetPressure(df)
        df = RemoveExcessivelyHighTankPressure(df)
        df = RemoveExcessivelyLargeTanks(df)
        df = CheckDuplicates(df) 
        df = RemoveFaultySensors(df, sensorNumber)
    df = DropUselessColumns(df)
    df = ChooseSensors(df, sensorNumber)
    return df