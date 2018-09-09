"""
Copyright 2017 Baris Akgun (baakgun@ku.edu.tr)

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimer in the documentation and/or other 
materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may 
be used to endorse or promote products derived from this software without specific 
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.

This software is intended for educational purposes only. 
"""
import numpy as np
import learners
import util
import data

if __name__ == "__main__":
  """
  In this homework, you are going to:
  - Extract normalized saturation histograms from images. See the SaturationHistogramExtractor class in data.py
  - Implement kNN learning and prediction. See the knnClassifier class in learners.py
  - Implement linear regression learning and prediction. See the LinearRegression class in learners.py
  
  This file is provided for you to test your implementations. You can:
  - Work on your features by only calling getDataForClassification(). When you are confident with your features, you can 
  set pickleEnabled to True to save and load the features efficiently.
  - Work on your kNN method by setting classification to True (default). If image features are be too hard to get started, create your own dummy data set and work on kNN.
  - Work on your linear regression method by setting regression to True. If you ridge regression slows you downmodify the regressors dictionary to temporarily remove it.
  
  When you run this function, it prints to standard output. In addition, it saves performance figures from the methods. Example outputs will be provided for you.
  Even though it is not required, I suggest that you inspect the resulting plots.
  
  We are going to use a different file and potentially different datasets to test your implementations.
  As long as you only change the part of the files that are designated and you match the outputs of this file, you should be fine.
  See the example-outputs folder.
  
  We are going to grade 3 things:
  - Your image features (data.py)
  - Your kNN implmentation (learners.py)
  - Your linear regression implementation (learners.py)
  
  Many other things that are implemented for you such as data loading and cross-validation
  I suggest you look at all the methods related to cross-validation implementation (util.py).
  Python pickle is also a nice feature to save and load arbitrary data structures (data.py).
  
  """

  #Set this to True when you are sure your feature extraction is working correctly
  pickleEnabled = False  

  #use these flags to work on a single problem
  classification = True
  regression = True
  
  #DO NOT MODIFY BELOW THIS LINE OTHER THAN TO DEBUG! OTHERWISE, YOUR OUTPUTS MIGHT BE DIFFERENT THAN THE PROVIDED EXAMPLES EVEN IF YOUR IMPLEMENTATIONS ARE CORRECT
  rSeed = 265341
  numCrossVal = 5

  if classification:
    #Classification
    print("--- Classification ---")
    print

    AllData, AllLabels = data.getDataForClassification(pickleEnabled=pickleEnabled)

    np.random.seed(rSeed) 
    kRange =  range(1,22,2)     
    knnParam, knnRes, knnStd = util.getParamAndRes(AllData, AllLabels, learners.knnClassifier, kRange, util.calculateClassificationError, numCrossVal)
    print "Selected k for kNN: ", knnParam
    print "kNN cross-validation average error for the selected k:" ,knnRes
    print "kNN cross-validation standard deviation of the error for the selected k:" ,knnStd

    print
    np.random.seed(rSeed)
    cRange = np.arange(1.0, 3.01, 0.2)
    cRange = np.power(10,cRange)
    lrParam, lrRes, lrStd = util.getParamAndRes(AllData, AllLabels, learners.LogisticRegressionClassifier, cRange, util.calculateClassificationError, numCrossVal, logTick = True)
    print "Selected regularizer (C) for logistic regression: ", lrParam
    print "Logreg cross-validation average error for the selected C:" ,lrRes
    print "Logreg cross-validation standard deviation of the error for the selected C:" ,lrStd

    print 
    if lrRes < knnRes:
      print "Logreg has a better cross-validation score than kNN with %.2f vs %.2f" % (lrRes, knnRes)
    elif lrRes > knnRes:
      print "kNN has a better cross-validation score than Logreg with %.2f vs %.2f" % (knnRes, lrRes)
    else:
      print "Both logreg and knn have the samet cross-validation score"

    print

  if regression:
    #Regression
    print("--- Regression ---")
    print

    dataNames = ['prostate','airfoil','hydrodynamics']
    Data, Target = data.getDataForRegression(dataNames)

    regressors = [learners.LinearRegression, learners.RidgeRegression] 
    paramSets = {learners.LinearRegression.name:[1], \
                 learners.RidgeRegression.name:np.arange(0, 16.01, 0.1)} 
    for name in dataNames:
      for method in regressors:
        np.random.seed(rSeed) 
        linParam, linRes, linStd = util.getParamAndRes(Data[name], Target[name],method, paramSets[method.name], util.calculateMeanL2Error, numCrossVal, "_"+name,)
        print "Selected parameter for " + method.name + " and " + name + " data: " + str(linParam)
        print method.name + " cross-validation average error for " + name + " data: " + str(linRes)
        print method.name + " cross-validation standard deviation of the error for " + name + " data: "  + str(linStd)
        print

    print "Selected parameter for Linear regression does not matter. Ridge regression with regularizer 0 is equivalent to linear regression. If a non-zero parameter is selected for ridge regression, than this implies it performed better than linear regression."
