import collections
from array import array

import dataClassifier
import util
import classificationMethod
import math
import numpy as np


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
  See the project description for the specifications of the Naive Bayes classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
    Test=None
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 2  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = True  # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
    Outside shell to call your method. Do not modify this method.
    """

        self.features = trainingData[0].keys()  # this could be useful for your code later...

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter
    that gives the best accuracy on the held-out validationData.

    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """

        "*** YOUR CODE HERE ***"
        row_count=0
        col_count=0
        grid=1

        trainingLabels = trainingLabels + validationLabels
        trainingData = trainingData + validationData
        P_Y_Count = collections.Counter(trainingLabels)
        global Test

        row_count = dataClassifier.DIGIT_DATUM_HEIGHT
        col_count = dataClassifier.DIGIT_DATUM_WIDTH

        for k in P_Y_Count.keys():
            P_Y_Count[k] = P_Y_Count[k] / (len(trainingLabels))
        sec = dict()

        for keys in P_Y_Count.keys():  # For every item we create a new dict
            sec[keys] = collections.defaultdict(list)  # Create the sec of default dictionary list

        for x, prob in P_Y_Count.items():
            first = list()
            for i, ptr in enumerate(trainingLabels):  # go through the traningLabels and check the indexs and append
                if x == ptr:  # Check the index
                    first.append(i)

            second = list()

            for i in first:  # Second is list that will contain training data based on labels
                second.append(trainingData[i])
            keys = list()
            for y in range(len(second)):  # Now we populate the dictionary with the correct label and the data
                a = np.array(list(second[y].values()))
                b = np.reshape(a, (row_count, col_count))
                key = list()
                for z in range(0, row_count, grid):
                    for y in range(0, col_count, grid):
                        key.append((b[z:z + grid, y:y + grid]))

                keys = list()
                for a in key:
                    keys.append(np.sum(a))
                for r, val in enumerate(keys):
                    sec[x][r].append(val)

        count = [a for a in P_Y_Count]  # Get the total count

        # for x in count:
        #  for k, ptr in second[0].items():
        #   sec[x][k[1]] = self.check(sec[x][k[1]])  # Get the probabilties for Naive Bayes

        for k, ptr in sec.items():
            x = ptr.keys()
            y = ptr.values()
            for i, j in zip((x), (y)):
                sec[k][i] = self.check(j)

        self.intial = P_Y_Count  # Update the P_Y_Count
        self.count = count  # Update the count
        self.sec = sec  # Update the second list with the training label and training data

    # util.raiseNotDefined()

    def check(self, out):
        prob = dict(collections.Counter(out))
        for k in prob.keys():
            prob[k] = prob[k] / float(len(out))
        return prob

    def classify(self, testData):
        """
    Classify the data based on the posterior distribution over labels.

    You shouldn't modify this method.
    """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        row = 0
        col = 0
        grid = 1
        global Test
        row = dataClassifier.DIGIT_DATUM_HEIGHT
        col = dataClassifier.DIGIT_DATUM_WIDTH
        a = np.array(list(datum.values()))
        b = np.reshape(a, (row, col))
        key = list()
        for z in range(0, row, grid):
            for y in range(0, col, grid):
                key.append((b[z:z + grid, y:y + grid]))
        keys = list()
        for a in key:
            keys.append(np.sum(a))

        n = dict()
        for x in self.count:
            probs = self.intial[x]  # Get the probabilty
            probs=math.log(probs)

            nf = self.sec.get(x)
            for k, ptr in enumerate(keys):
                # Get the data we need from the sec dict
                if nf.get(k).get(ptr) == None:
                    probs = probs + math.log(0.000001)
                    continue
                else:
                    p = nf.get(k).get(ptr)
                    probs = probs + math.log(p)  # Calculate the probability

            logJoint[x] = probs  # Add the new probability back to the log Joint list
        # util.raiseNotDefined()
        m = max(logJoint.values())
        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2)
    """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresOdds




