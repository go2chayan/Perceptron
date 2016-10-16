# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:39:44 2014
Programming Assignment 2: Perceptron Algorithm
@author: Md. Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""
import random
import numpy as np
from sys import stdout

trainN = 348
testN = 42

# Read each line and convert into 2D array
dictionary = {'democrat':1,'republican':-1,'y':1,'n':-1,'?':0}
input_file = open('voting2.dat')
lines = input_file.readlines()
allData = [line.strip().split(',') for line in lines if (\
    line.startswith('republican') or \
    line.startswith('democrat'))]
for i in range(np.size(allData,0)):
    for j in range(np.size(allData,1)):
        allData[i][j] = dictionary[allData[i][j]]
allData = np.array(allData)

accSum = 0
maxIter = 1000

# Random shuffle and repeat for calculating average accuracy
for i in range(0,10):
    # Create training, development and test set by random sampling
    allIdx = set(range(np.size(allData,0)))
    trainingSet = random.sample(allIdx,trainN)
    testSet = random.sample(allIdx.difference(trainingSet),testN)
    X_tr = np.hstack((allData[trainingSet,1:],np.ones([trainN,1])))
    Y_tr = allData[trainingSet,0][None].T
    X_te = np.hstack((allData[testSet,1:],np.ones([testN,1])))
    Y_te = allData[testSet,0][None].T
            
    # Model building on training set using Perceptron Algorithm
    print 'Building model and testing (',i+1,' / 10 ) ...',
    w = np.ones([X_tr.shape[1],1])
    iter = 0
    alpha = 0.4
    while (sum(np.sign(X_tr.dot(w)) != Y_tr) > 0) and (iter < maxIter):                    
        j = 0
        for aRow in X_tr:
            if(np.sign(aRow.dot(w))!=Y_tr[j,0]):  # if prediction is incorrect
                w = w + alpha*Y_tr[j,0]*aRow[None].T    # update w with y(n)X(n,:)
            j = j + 1
        iter = iter + 1
        stdout.flush()
    
    # Apply on test data and calculate the final accuracy
    Y_te_hat = np.sign(X_te.dot(w))
    acc = sum(Y_te_hat == Y_te)/float(testN)
    accSum = accSum + acc
    print 'Finished. Accuracy = %0.2f' % acc

print 'Average Accuracy =', accSum/10., '(may change in each run due to', \
    'random sample)'
