#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# BLFore running this script, you are supposed to have the features by running run.feature.sh

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal.
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups.

# Paths to different tools;
map_path=/home/ubuntu/tools/mAP
export PATH=$map_path:$PATH

echo ""
echo "#####################################"
echo "#       MED with LF Features       #"
echo "#####################################"
mkdir -p LF_pred

feat_dim_LF=50
# iterate over the events
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python train_LF.py $event "LF_vector/" $feat_dim_LF LF_pred/svm.$event.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python test_LF.py LF_pred/svm.$event.model "LF_vector/" $feat_dim_LF LF_pred/${event}_val_LF.lst || exit 1;
  # compute the average precision by calling the mAP package
  ap ${event}_val_label LF_pred/${event}_val_LF.lst
done
