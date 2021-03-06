#!/bin/python

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    fread = open('all_val.lst',"r")
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

        if i == 0:
            X_test = X
            i = 1
        else:
            X_test = np.vstack((X_test, X))

    print (X_test)
    clf = cPickle.load(open(model_file,"rb"))
    print (clf.predict(X_test))

    np.savetxt(output_file, clf.decision_function(X_test))
