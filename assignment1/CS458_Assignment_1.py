#
# file  CS490_Assignment_1.py
# brief Purdue University Fall 2022 CS490 robotics Assignment 1 -
#       Gaussian Discriminant Analysis
# date  2022-09-01
#

#you can only import modules listed in the handout
import sys


#hand label region related functions
#**************************************************************************************************
#hand label pos/neg region for training data
#write regions to files, this function should not return anything
def label_training_dataset(training_path, region_path):
    pass

#hand label pos region for testing data
#write regions to files, this function should not return anything
def label_testing_dataset(training_path, region_path):
    pass
#**************************************************************************************************


#import labeled regions related functions
#**************************************************************************************************
#import pre hand labeled region for trainning data
def import_pre_labeled_training(training_path, region_path):
    features, labels = None, None
    return features, labels

#import per hand labeled region for testing data
def import_pre_labeled_testing(testing_path, region_path):
    features, labels = None, None
    return features, labels
#**************************************************************************************************


#main GDA training functions
#**************************************************************************************************
def train_GDA_common_variance(features, labels):
    prior, mu, cov = None, None, None
    return prior, mu, cov

def train_GDA_variable_variance(features, labels):
    prior, mu, cov = None, None, None
    return prior, mu, cov
#**************************************************************************************************


#GDA testing and accuracy analyis functions
#**************************************************************************************************
#assign labels using trained GDA parameters for testing features
def predict(testing_features, theta, mu, cov):
    predicted_labels = None
    return predicted_labels

#print precision/call for both classes to console
#
#example console printout:
#GDA with common variance:
#precision of label 0: xx.xx%
#recall of label 0:    xx.xx%
#precision of label 1: xx.xx%
#recall of label 1:    xx.xx%
#GDA with variable variance:
#precision of label 0: xx.xx%
#recall of label 0:    xx.xx%
#precision of label 1: xx.xx%
#recall of label 1:    xx.xx%
#
def accuracy_analysis(predicted_labels, ground_truth_labels):
    pass
#**************************************************************************************************
    

if __name__ == '__main__':
    #Please read this block before coding
    #**********************************************************************************************
    #caution: when you submit this file, make sure the main function is unchanged otherwise your
    #         grade will be affected because the grading script is designed based on the current
    #         main function
    #
    #         Also, do not print unnecessary values other than the accuracy analysis in the console

    #Labeling during runtime can be very time-consuming during the debugging phase. 
    #Also, it is hard to ensure the labelings are consistent during each testing run. 
    #Thus, we do this in separate stages.
    #First, implement all the functions and uncomment the three lines in the data loader block.
    #Then, revert the main function back to what it is used to be and start implementing the rest
    #**********************************************************************************************


    #data loader used to generate your labeling
    #ideally this block should only be called once
    #**********************************************************************************************
    #label_training_dataset('trainset', 'train_region')
    #label_testing_dataset('testset', 'test_region')
    #sys.exit(1)
    #**********************************************************************************************


    #import your generated labels from saved data
    #**********************************************************************************************
    training_features, training_labels = import_pre_labeled_training('trainset', 'train_region')
    testing_features, ground_truth_labels = import_pre_labeled_testing('testset', 'test_region')
    #**********************************************************************************************


    #GDA with common varianve
    #**********************************************************************************************
    prior, mu, cov = train_GDA_common_variance(training_features, training_labels)

    predicted_labels = predict(testing_features, prior, mu, cov)

    accuracy_analysis(predicted_labels, ground_truth_labels)
    #**********************************************************************************************


    #GDA with variable variance
    #**********************************************************************************************
    prior, mu, cov = train_GDA_variable_variance(training_features, training_labels)

    predicted_labels = predict(testing_features, prior, mu, cov)

    accuracy_analysis(predicted_labels, ground_truth_labels)
    #**********************************************************************************************
