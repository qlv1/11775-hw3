#!/bin/python

import numpy as np
import os
from sklearn.svm.classes import SVC, OneClassSVM
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import laplacian_kernel
import cPickle
import sys
import random

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    fread = open('all_trn.lst',"r")
    dimension_i = [0, 2048, 100, 8852]

    i = 0
    for line in fread.readlines():
        mfcc_name, label = line.split(' ')
        mfcc_path_1 = '../features/resnet50/' + mfcc_name.replace('\n','') + ".npy"
        mfcc_path_2 = '../hw1_code/mfcc_vector/' + mfcc_name.replace('\n','') + ".mfcc.csv"
        mfcc_path_3 = '../hw1_code/asr_vector/' + mfcc_name.replace('\n','') + ".asr.csv"

        X = np.zeros(2048 + 100 + 8852)
        for j, path_i in enumerate([mfcc_path_1, mfcc_path_2, mfcc_path_3]):
            if os.path.exists(path_i) == True:
                if j == 0:
                    X_i = np.load(path_i)
                else:
                    X_i = np.genfromtxt(path_i)
                X[dimension_i[j] : dimension_i[j] + dimension_i[j+1]] = X_i

        Y_label = label.replace('\n','')
        if Y_label != 'NULL' or random.random() > 0:
            if Y_label == event_name:
                Y = 1
            else:
                Y = 0

            if i == 0:
                X_all = X
                Y_all = Y
                i = 1
            else:
                X_all = np.vstack((X_all, X))
                Y_all = np.append(Y_all, Y)
                i += 1

    clf = SVC(kernel = laplacian_kernel)

    features = ['.cnn.', '.mfcc.', '.asr.']
    X_score = np.zeros((len(X_all), 3))


    j = 0
    for j, feature in enumerate(features):
        X = X_all[:, dimension_i[j] : dimension_i[j] + dimension_i[j+1]]
        clf.fit(X, Y_all)
        print ('saving scores...')
        X_score[:, j] = clf.decision_function(X)
        # print (X_score[:, j])
        np.save(event_name + feature + 'score', X_score[:, j])
        cPickle.dump(clf, open(output_file + feature + 'score', "wb"))


    clf = SVC(kernel = 'linear')
    clf.fit(X_score, Y_all)

    fread.close()

    cPickle.dump(clf, open(output_file, "wb"))

    print 'SVM trained successfully for event %s!' % (event_name)
