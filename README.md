# hb-on-cnn-features-for-scenes
This repository contains a suite of MATLAB functions that build a hierarchical bayes classifier working in the space of features produced by convolutional neural network for the images of scenes. This work is guided by the paper "One-Shot Learning with a Hierarchical
Nonparametric Bayesian Model" by Ruslan Salakhutdinov, Josh Tenenbaum and Antonio Torralba (2011).

You are required to have MatConvNet installed in order to create your own datasets to train the model. You can download the scene images dataset here (provided as part of 6.869 course at MIT):
http://6.869.csail.mit.edu/fa15/challenge/data.tar.gz

This repository contains the trained network model and the part of the dataset that has been used (in case you just want to run the code without having to install MatConvNet). You might have to adjust a few parameters in the code to get everything up and running but generally it should be ok.
