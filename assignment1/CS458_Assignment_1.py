#
# file  CS490_Assignment_1.py
# brief Purdue University Fall 2022 CS490 robotics Assignment 1 -
#       Gaussian Discriminant Analysis
# date  2022-09-01
#

#you can only import modules listed in the handout
import math
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from roipoly import RoiPoly


#hand label region related functions
#**************************************************************************************************
#hand label pos/neg region for training data
#write regions to files, this function should not return anything
def label_training_dataset(training_path, region_path):
    if os.path.isdir(training_path):
        os.makedirs(region_path, exist_ok=True)
    else:
        print(f"The directory {training_path} does not exist!")
        return

    for filename in os.listdir(training_path):
        filepath = os.path.join(training_path, filename)
        if os.path.isfile(filepath):
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            pos_filepath = os.path.join(region_path, base_name + "-pos.npy")
            neg_filepath = os.path.join(region_path, base_name + "-neg.npy")
            if os.path.isfile(pos_filepath) and os.path.isfile(neg_filepath):
                # Skip labeling images with positive and negative regions saved
                print(f"Skipped labeling {filepath} - positive/negative regions already saved")
                continue

            img = plt.imread(filepath)

            plt.imshow(img)
            plt.title(f"{filename} - draw positive region (barrel)")
            pos_roi = RoiPoly(color='g')

            plt.imshow(img)
            plt.title(f"{filename} - draw negative region (similar size)")
            pos_roi.display_roi()
            neg_roi = RoiPoly(color='r')

            # Display both positive and negative regions
            plt.imshow(img)
            plt.title(f"{filename} - close window to advance to next image")
            pos_roi.display_roi()
            neg_roi.display_roi()
            plt.show()

            # Save positive and negative region masks
            pos_mask = pos_roi.get_mask(img[:, :, 0])
            neg_mask = neg_roi.get_mask(img[:, :, 0])
            np.save(pos_filepath, pos_mask)
            np.save(neg_filepath, neg_mask)


#hand label pos region for testing data
#write regions to files, this function should not return anything
def label_testing_dataset(training_path, region_path):
    if os.path.isdir(training_path):
        os.makedirs(region_path, exist_ok=True)
    else:
        print(f"The directory {training_path} does not exist!")
        return

    for filename in os.listdir(training_path):
        filepath = os.path.join(training_path, filename)
        if os.path.isfile(filepath):
            base_name = os.path.splitext(os.path.basename(filename))[0]
            pos_filepath = os.path.join(region_path, base_name + "-pos.npy")
            if os.path.isfile(pos_filepath):
                # Skip labeling images with positive region saved
                print(f"Skipped labeling {filepath} - positive region already saved")
                continue

            img = plt.imread(filepath)

            plt.imshow(img)
            plt.title(f"{filename} - draw positive region (barrel)")
            pos_roi = RoiPoly(color='g')

            # Display positive region
            plt.imshow(img)
            plt.title(f"{filename} - close window to advance to next image")
            pos_roi.display_roi()
            plt.show()

            # Save positive region mask
            pos_mask = pos_roi.get_mask(img[:, :, 0])
            np.save(pos_filepath, pos_mask)
#**************************************************************************************************


#import labeled regions related functions
#**************************************************************************************************
#import pre hand labeled region for trainning data
def import_pre_labeled_training(training_path, region_path):
    features, labels = [], []

    for filename in os.listdir(training_path):
        filepath = os.path.join(training_path, filename)
        if os.path.isfile(filepath):
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            pos_filepath = os.path.join(region_path, base_name + "-pos.npy")
            neg_filepath = os.path.join(region_path, base_name + "-neg.npy")
            if not (os.path.isfile(pos_filepath) and os.path.isfile(neg_filepath)):
                # Skip images without positive and negative regions saved
                print(f"Skipped importing {filepath} - positive/negative regions missing")
                continue

            # Read in image and positive/negative regions
            img = plt.imread(filepath) # shape: (height, width, channels)
            pos_mask = np.load(pos_filepath) # shape: (height, width)
            neg_mask = np.load(neg_filepath)

            # Flatten image pixels and positive/negative regions
            channels = img.shape[-1] # 3 for RGB
            pixels_vec = img.reshape(-1, channels) # shape: (height * width, channels)
            pos_vec = pos_mask.flatten()
            neg_vec = neg_mask.flatten()

            # Label positive pixels with 1, negative pixels with 0
            for pixel, pos, neg in zip(pixels_vec, pos_vec, neg_vec):
                if pos:
                    features.append(pixel)
                    labels.append(1)
                if neg:
                    features.append(pixel)
                    labels.append(0)
    
    return np.asarray(features), np.asarray(labels)


#import per hand labeled region for testing data
def import_pre_labeled_testing(testing_path, region_path):
    features, labels = [], []

    for filename in os.listdir(testing_path):
        filepath = os.path.join(testing_path, filename)
        if os.path.isfile(filepath):
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            pos_filepath = os.path.join(region_path, base_name + "-pos.npy")
            if not os.path.isfile(pos_filepath):
                # Skip images without positive region saved
                print(f"Skipped importing {filepath} - positive region missing")
                continue

            # Read in image and positive region
            img = plt.imread(filepath) # shape: (height, width, channels)
            pos_mask = np.load(pos_filepath) # shape: (height, width)

            # Flatten image pixels and positive region
            channels = img.shape[-1] # 3 for RGB
            pixels_vec = img.reshape(-1, channels) # shape: (height * width, channels)
            pos_vec = pos_mask.flatten()

            # Label positive pixels with 1, negative pixels (everything else) with 0
            for pixel, pos in zip(pixels_vec, pos_vec):
                if pos:
                    features.append(pixel)
                    labels.append(1)
                else:
                    features.append(pixel)
                    labels.append(0)
    
    return np.asarray(features), np.asarray(labels)
#**************************************************************************************************


#main GDA training functions
#**************************************************************************************************
def train_GDA_common_variance(features, labels):
    x = np.asarray(features)
    y = np.asarray(labels)
    n, d = x.shape
    if n == 0:
        prior = np.array([0.5, 0.5])
        mu = np.zeros((2, d))
        cov = np.identity(d)
        return prior, mu, (cov, cov)

    x0 = x[y == 0]
    x1 = x[y == 1]

    # Compute priors
    n0 = x0.shape[0]
    n1 = x1.shape[0]
    p0 = n0 / n
    p1 = n1 / n
    prior = np.array([p0, p1])

    # Compute means (along columns)
    mu0 = np.mean(x0, axis=0) if n0 > 0 else np.zeroes(d)
    mu1 = np.mean(x1, axis=0) if n1 > 0 else np.zeroes(d)
    mu = np.array([mu0, mu1])

    # Common covariance matrix
    cov = np.zeros((d, d))
    if n0 > 0:
        # x0: (n, d) - n rows, each with d elements
        # mu0: (d) - single row
        # xdiff0: x0 with mu0 subtracted from each row
        #   - each row now represents some (xi - mu0)
        xdiff0 = x0 - mu0
        cov += xdiff0.T @ xdiff0
    if n1 > 0:
        xdiff1 = x1 - mu1
        cov += xdiff1.T @ xdiff1
    cov = cov / (n - 1)

    return prior, mu, (cov, cov)

def train_GDA_variable_variance(features, labels):
    x = np.asarray(features)
    y = np.asarray(labels)
    n, d = x.shape
    if n == 0:
        prior = np.array([0.5, 0.5])
        mu = np.zeros((2, d))
        cov = np.identity(d)
        return prior, mu, (cov, cov)

    x0 = x[y == 0]
    x1 = x[y == 1]

    # Compute priors
    n0 = x0.shape[0]
    n1 = x1.shape[0]
    p0 = n0 / n
    p1 = n1 / n
    prior = np.array([p0, p1])

    # Compute means (along columns)
    mu0 = np.mean(x0, axis=0) if n0 > 0 else np.zeroes(d)
    mu1 = np.mean(x1, axis=0) if n1 > 0 else np.zeroes(d)
    mu = np.vstack([mu0, mu1])

    cov0 = np.identity(d)
    cov1 = np.identity(d)
    if n0 > 0:
        xdiff0 = x0 - mu0
        cov0 = (xdiff0.T @ xdiff0) / (n0 - 1)
    if n1 > 0:
        xdiff1 = x1 - mu1
        cov1 = (xdiff1.T @ xdiff1) / (n1 - 1)

    return prior, mu, (cov0, cov1)
#**************************************************************************************************


#GDA testing and accuracy analyis functions
#**************************************************************************************************
#assign labels using trained GDA parameters for testing features
def predict(testing_features, prior, mu, cov):
    x = testing_features
    p0, p1 = prior[0], prior[1]
    mu0, mu1 = mu[0], mu[1]
    cov0, cov1 = cov

    _, logDet0 = np.linalg.slogdet(cov0)
    _, logDet1 = np.linalg.slogdet(cov1)

    invCov0 = np.linalg.inv(cov0)
    invCov1 = np.linalg.inv(cov1)

    logPrior0 = math.log(p0)
    logPrior1 = math.log(p1)

    # x: (n, d) - n rows, each with d elements
    # mu0: (d) - single row
    # xdiff0: x with mu0 subtracted from each row
    #   - each row now represents some (xi - mu0)
    xdiff0 = x - mu0
    xdiff1 = x - mu1

    # We wish to compute the term 
    #       -0.5 * [ (x-mu)^T @ invCov @ (x-mu) ]
    # for each pixel x.
    #
    # In the above expression, x (hence x-mu) is a column vector, but in our
    # code, (x-mu) corresponds to rows in xdiff. Hence
    #       xdiff @ invCov
    # gives a matrix where for each row (x-mu) in xdiff, the corresponding row
    # in the resulting matrix is
    #       (x-mu)^T @ invCov
    # Now, taking the dot product of this and (x-mu) gives exactly
    #       (x-mu)^T @ invCov @ (x-mu)
    # So, do component-wise multiplication (*) with xdiff (remember, (x-mu) are
    # rows in xdiff), then sum across each row. The result is an n-dim vector
    # where each entry is
    #       (x-mu)^T @ invCov @ (x-mu)
    # for the corresponding row x, as desired.
    var0 = -0.5 * np.sum((xdiff0 @ invCov0) * xdiff0, axis=1)
    var1 = -0.5 * np.sum((xdiff1 @ invCov1) * xdiff1, axis=1)

    # Add scalar term (-0.5*logDet + logPrior) to every entry of above vector
    delta0 = var0 - 0.5 * logDet0 + logPrior0
    delta1 = var1 - 0.5 * logDet1 + logPrior1

    predicted_labels = np.where(delta1 > delta0, 1, 0)
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
    # label_training_dataset('trainset', 'train_region')
    # label_testing_dataset('testset', 'test_region')
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
