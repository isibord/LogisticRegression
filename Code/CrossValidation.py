import Assignment1Support
import EvaluationsStub
import collections
import math
import numpy as np

### UPDATE this path for your environment
kDataPath = "..\\Data\\SMSSpamCollection"

(xRaw, yRaw) = Assignment1Support.LoadRawData(kDataPath)

(xTrainRaw, yTrainRaw, xTestRaw, yTestRaw) = Assignment1Support.TrainTestSplit(xRaw, yRaw)

(xTrain, xTest) = Assignment1Support.Featurize(xTrainRaw, xTestRaw)
yTrain = yTrainRaw
yTest = yTestRaw

import LogisticRegressionModel_NumPy
model = LogisticRegressionModel_NumPy.LogisticRegressionModel_NumPy()

numFolds = 5
totalCorrect = 0

for i in range(numFolds):
    (foldTrainX, foldTrainY)  = Assignment1Support.GetAllDataExceptFold(xTrain, yTrain, i, numFolds)
    (foldValidationX, foldValidationY) = Assignment1Support.GetDataInFold(xTrain, yTrain, i, numFolds)

    # do feature engineering/selection on foldTrainX, foldTrainY

    
    xTrain_np = np.asarray(foldTrainX)
    yTrain_np = np.asarray(foldTrainY)
    xTest_np = np.asarray(foldValidationX)
    yTest_np = np.asarray(foldValidationY)

    model.fit(xTrain_np, yTrain_np, iterations=50000, step=0.01)

    yTestPredicted = model.predict(xTest_np)

    #print("%d, %f, %f, %f" % (50000, model.weights[1], model.loss(xTest_np, yTest_np), EvaluationsStub.Accuracy(yTest_np, yTestPredicted)))

    #EvaluationsStub.ExecuteAll(foldValidationY, yTestPredicted)

    totalCorrect += EvaluationsStub.CountCorrect(yTestPredicted, foldValidationY)
     
accuracy = totalCorrect / len(xTrain)

upper = accuracy + 1.96 * math.sqrt( (accuracy * (1 - accuracy) ) / len(xTrain) )
lower  = accuracy - 1.96 * math.sqrt( (accuracy * (1 - accuracy) ) / len(xTrain) )

print("acc: ", accuracy)
print("upper: ", upper)
print("lower: ", lower)
