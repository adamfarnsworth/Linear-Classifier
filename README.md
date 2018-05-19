# Linear classifier

## Training (using the training data set):
1. Computes the centroid of each class (A, B, and C).
2. Constructs a discriminant function between each pair of classes (A/B, B/C, and
A/C), halfway between the two centroids and orthogonal to the line connecting
the two centroids. This is the “basic linear classifier”.  

## Testing (using the testing data set):
1. For each instance, the discriminant function to decides “A or B” and then
(depending on that answer) to decides “A or C” or “B or C.” (Ties give
priority to class A, then B, then C.)
2. Tracks the true positives, true negatives, false positives, and false negatives.  

## Output
The program returns a dictionary of averages of the true positive rate, the
false positive rate, the error rate, the accuracy, and the precision – e.g., as shown here:  

 ```python
% print(run_train_test(training_input, testing_input)
{
“tpr”: 0.80 # true positive rate
“fpr”: 0.27 # false positive rate
“error_rate”: 0.44
“accuracy”: 0.60
“precision”: 0.90
}
 ```
