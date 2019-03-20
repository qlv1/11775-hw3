#!/bin/python

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video
i = 1

for file in ['P001', 'P002', 'P003']:
    # pred = np.loadtxt('EF_pred/' + file + '_test_EF.lst')
    # pred = np.loadtxt('EF_pred/' + file + '_test_EF_submit.lst')
    pred = np.loadtxt('DF_pred_submit/' + file + '_test_DF.lst')
    if i == 1:
        pred_all = np.zeros((len(pred), 4))
        # pred_all[:, 0] = -1.45
        pred_all[:, 0] = -1.015
    pred_all[:, i] = pred
    i += 1


pred_final = np.zeros(len(pred)).astype(int)

for i in range(len(pred)):
    pred_final[i] = np.argmax(pred_all[i])

print (pred_final)
np.savetxt('pred_new_DF.csv', pred_final)
# np.savetxt('pred_new_EF.csv', pred_final)
# np.savetxt('pred_new_EF_submit.csv', pred_final)
