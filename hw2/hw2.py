import numpy as np
import collections

# computing precision TP/Phat
def precX(TP,Phat):
    return float(float(TP) / float(Phat))

# computing accuracy rates (tp + tn)/(p+n)
def accRateX(TP, TN, P, N):
    return float(float(TP + TN) / float(P + N))

# computing false positive rates FP/N
def fprX(FP,N):
    return float(float(FP) / float(N))

# compute true positive rate TP/P
def tprX(TP,P):
    return float(float(TP) / float(P))

# compute error rate (fp+fn)/(p+n)
def errRateX(FP,FN,P,N):
    return float(float(FP + FN) / float(P + N))

def findScalar(x,y):
   return x - y

def findCenter(x,y):
    return (x + y) / 2

#input: point -> value to be checked
#       x,y -> A/B, A/C, B/C
#
#return: boolean -> if the point is x or y
def valueGreater(point, x, y):
    center = findCenter(x,y)
    scalar = findScalar(x,y)
    
    total = 0
    # a(x-x1) + ...  + n(z - z1)
    for i in range(0, point.size):
        total += scalar[i] * (point[i] - center[i])
    
    # check if above or below, which determines if its x or y
    if total >= 0:
        return True   
    return False

def run_train_test(training_input, testing_input):

    # array of arrays, pretty much a matrix
    training = np.array([np.array(i) for i in training_input])
    testing = np.array([np.array(i) for i in testing_input])

    # our loop sizes
    dimensionOfData = training[0][0]
    examplesOfA = int(training[0][1])
    examplesOfB = int(training[0][2])
    examplesOfC = int(training[0][3])

    # centroids for our classes
    centroidA = np.zeros(int(dimensionOfData))
    centroidB = np.zeros(int(dimensionOfData))
    centroidC = np.zeros(int(dimensionOfData))

    # adding all values in A, then averaging them
    for i in range(1, examplesOfA + 1):
        centroidA += training[i]
    centroidA = centroidA / examplesOfA

    # adding all values in B, then averaging them
    for i in range(1, examplesOfB + 1):
        centroidB += training[examplesOfA + i]
    centroidB = centroidB / examplesOfB

    # adding all values in C, then averaging them
    for i in range(1, examplesOfC + 1):
        centroidC += training[examplesOfA + examplesOfB + i]
    centroidC = centroidC / examplesOfC

    # count for how many times we think its that class
    countA = 0
    countB = 0
    countC = 0
    # count for how may times we were right
    tpA = 0
    tpB = 0
    tpC = 0
    testCount = 0

    testsOfA = int(testing[0][1])
    testsOfB = int(testing[0][2])
    testsOfC = int(testing[0][3])
    # running testing data in model
    for i in testing[1:]:
        testCount +=1

        if valueGreater(i, centroidA, centroidB): #A/B
            if valueGreater(i, centroidA, centroidC): #A/C
                countA += 1
                if testCount <= testsOfA:
                    tpA +=1
            else:
                countC += 1
                if (testsOfB + testsOfA) < testCount:
                    tpC +=1
        elif valueGreater(i, centroidB, centroidC): #B/C
            countB += 1
            if testsOfA < testCount <= (testsOfA + testsOfB):
                tpB +=1
        else:
            countC += 1
            if (testsOfA + testsOfB) < testCount:
                tpC +=1


    
    # computing false negative
    fnA = testsOfA - tpA
    fnB = testsOfB - tpB
    fnC = testsOfC - tpC


    # computing true negatives
    tnA = countB + countC - fnA
    tnB = countA + countC - fnB
    tnC = countA + countB - fnC


    # computing false positives
    fpA = countA - tpA
    fpB = countB - tpB
    fpC = countC - tpC


    # computing true positve rates TP/P
    tprA = tprX(tpA, testsOfA)
    tprB = tprX(tpB, testsOfB)
    tprC = tprX(tpC, testsOfC)


    # computing false positive rates FP/N
    fprA = fprX(fpA, testsOfB + testsOfC)
    fprB = fprX(fpB, testsOfA + testsOfC)
    fprC = fprX(fpC, testsOfA + testsOfB)
    
    # computing error rates (fp+fn)/(p+n)
    errRateA = errRateX(fpA,fnA,testsOfA,testsOfB + testsOfC)
    errRateB = errRateX(fpB,fnB,testsOfB,testsOfA + testsOfC)
    errRateC = errRateX(fpC,fnC,testsOfC,testsOfA + testsOfB) 
    
    # computing accuracy rates (tp + tn)/(p+n)
    accRateA = accRateX(tpA, tnA, testsOfA, testsOfB + testsOfC)
    accRateB = accRateX(tpB, tnB, testsOfB, testsOfA + testsOfC)
    accRateC = accRateX(tpC, tnC, testsOfC, testsOfA + testsOfB)

    # computing precision TP/Phat where Phat = TP + FP
    precA = precX(tpA, tpA + fpA)
    precB = precX(tpB, tpB + fpB)
    precC = precX(tpC, tpC + fpC)

    tpr = float((tprA + tprB + tprC) / 3)
    fpr = float((fprA + fprB + fprC) / 3)
    error_rate = float((errRateA + errRateB + errRateC) / 3)
    accuracy = float((accRateA + accRateB + accRateC) / 3)
    precision = float((precA + precB + precC) / 3)
    
    testDic = {'tpr': tpr, 'fpr': fpr,'error_rate': error_rate,'accuracy': accuracy,'precision': precision}
    
    print("TN", tnA, tnB, tnC)
    print("FP", fpA, fpB, fpC)
    print("FN", fnA, fnB, fnC)
    print("TP", tpA, tpB, tpC)
    print(testDic)
    return testDic


    # TODO: IMPLEMENT
    pass


#######
# The following functions are provided for you to test your classifier.
######
def parse_file(filename):
    """
    This function is provided to you as an example of the preprocessing we do
    prior to calling run_train_test
    """
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw2.py [training file path] [testing file path]
    """
    import sys
    training_input = parse_file(sys.argv[1])
    #"./data/training1.txt"
    
    testing_input = parse_file(sys.argv[2])
    #"./data/testing1.txt"
    

    run_train_test(training_input, testing_input)