import numpy as np
import collections

def findScalar(x,y):
   return x-y

def findCenter(x,y):
    return (x+y)/2

#input: point -> value to be checked
#       x,y -> A/B, A/C, B/C
#
#return: boolean -> if the point is x or y
def valueGreater(point, x, y):
    center = findCenter(x,y)
    scalar = findScalar(x,y)
    
    total = 0
    # a(x-x1) + ... + n(z - z1)
    for i in range(0, point.size):
        total += scalar[i]*(point[i] - center[i])
    
    # check if above or below, which determines if its x or yx
    if total > 0:
        return True   
    return False

def run_train_test(training_input, testing_input):

    # array of arrays, pretty much a matrix
    training = np.array([np.array(i) for i in training_input])
    testing = np.array([np.array(i) for i in testing_input])

    # our loop sizes
    dimensionOfData = training[0][0]
    examplesOfA = training[0][1]
    examplesOfB = training[0][2]
    examplesOfC = training[0][3]

    # centroids for our classes
    centroidA = np.zeros(dimensionOfData)
    centroidB = np.zeros(dimensionOfData)
    centroidC = np.zeros(dimensionOfData)

    # adding all values in A, then averaging them
    for i in range(1, examplesOfA + 1):
        centroidA += training[i]
    centroidA = centroidA/examplesOfA

    # adding all values in B, then averaging them
    for i in range(1, examplesOfB + 1):
        centroidB += training[examplesOfA + i]
    centroidB = centroidB/examplesOfB

    # adding all values in C, then averaging them
    for i in range(1, examplesOfC + 1):
        centroidC += training[examplesOfA + examplesOfB + i]
    centroidC = centroidC/examplesOfC

    if valueGreater(training[3], training[3], training[3]):
        print("hi")


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


    #    """
    #Implement the training and testing procedure here. You are permitted
    #to use additional functions but DO NOT change this function definition. 
    #You are you are permitted to use the numpy library but you must write 
    #your own code for the linear classifier. 

    #Inputs:
    #    training_input: list form of the training file
    #        e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
    #    testing_input: list form of the testing file

    #Output:
    #    Dictionary of result values 

    #    IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
    #    Example:
    #        return {
    #            "tpr": true_positive_rate,
    #            "fpr": false_positive_rate,
    #            "error_rate": error_rate,
    #            "accuracy": accuracy,
    #            "precision": precision
    #        }
    #"""