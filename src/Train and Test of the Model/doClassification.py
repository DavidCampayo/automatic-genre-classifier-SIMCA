#IMPORTS
import sys
import os
import csv
import numpy
import random
import weka.core.jvm as jvm
import weka.core.converters as conv
from weka.classifiers import Classifier, Evaluation


#https://github.com/fracpete/python-weka-wrapper-examples/blob/master/src/wekaexamples/classifiers/train_test_split.py

#INFO
print ("--------------------------------------------------------------------Info-----------------------------------------------------------------")
print ("This is a script that makes classificaton using each train ant test fold for each genre.                                                ")
print ("                                                                                                                                         ")
print ("David Campayo, Final Degree Project, Audiovisual Systems Engineering, 2018                                                               ")
print ("-----------------------------------------------------------------------------------------------------------------------------------------")
print ("                                                                                                                                         ")


#READING COMMAND LINE

if len(sys.argv) < 3:
    print ("----------------------------- How to use it-------------------------------")
    print ("                                                                 ")
    print ("Declaration:            -> doClassfication.py path\\to\\data\\folder path\\to\\data\\saving classification method")
    print ("Classification Methods: - SMO ")
    print ("                        - J48 ")
    print ("                        - IBk ")
    sys.exit(-1)

#DECLARATION DATA
trainTestFoldGenreDataFolder = sys.argv[1]
smoDataSaving = sys.argv[2]
classificationMethod = sys.argv[3]

#1. Searching Files

genre = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
trainTestFoldGenreDataList = os.listdir(trainTestFoldGenreDataFolder)

#print trainTestFoldGenreDataList

print ("This is your train fold genre data folder:         ")
print ("                                                   ")
for trainfile in trainTestFoldGenreDataList:
    for foldtrain in range(0,len(trainTestFoldGenreDataList)):
        for genreitem in range(0,len(genre)):
            if trainfile.startswith("data_fold"+str(foldtrain)+"_train_PCA_"+genre[genreitem]):
                print trainfile
                print ("                                           ")


print ("******************************************         ")
print ("******************************************         ")
print ("                                                   ")

print ("This is your test fold genre data folder:         ")
print ("                                                  ")
for testfile in trainTestFoldGenreDataList:
    for foldtest in range(0,len(trainTestFoldGenreDataList)):
        for genreitem in range(0,len(genre)):
            if testfile.startswith("data_fold"+str(foldtest)+"_test_PCA_"+genre[genreitem]):
                print testfile
                print ("                                           ")


if classificationMethod == "SMO":
    jvm.start()
    print "--------------------CLASSIFIER: S M O ------------------------"
    print "                                                   "
    print "----------------------FOLD 0-----------------------"
    print "                                                   "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())


    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.functions.SMO")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    jvm.stop()


if classificationMethod == "J48":
    jvm.start()
    print "-------------------- CLASSIFIER: J 48 ------------------------"
    print "                                                   "
    print "----------------------FOLD 0-----------------------"
    print "                                                   "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())


    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

jvm.stop()

if classificationMethod == "IBk":
    jvm.start()
    

    print "--------------------CLASSIFIER: IBk ------------------------"
    print "                                                   "
    print "----------------------FOLD 0-----------------------"
    print "                                                   "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_blues.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_blues.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_classical.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_classical.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_country.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_country.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())


    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_disco.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_disco.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_hiphop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_hiphop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_jazz.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_jazz.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_metal.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_metal.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_pop.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_pop.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_reggae.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_reggae.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 0----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold0_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 1----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold1_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 2----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold2_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 3----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold3_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 4----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold4_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 5----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold5_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 6----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold6_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 7----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold7_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 8----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold8_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    print "----------------------FOLD 9----------------------"
    print "                                                  "
    # generate train/test split of randomized data
    test = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_test_PCA_rock.csv")
    test.class_is_last()
    train = conv.load_any_file("C:\Users\David\Desktop\TFG_2\\6_AllGenres_Kfold_Train_Test_PCA\\data_fold9_train_PCA_rock.csv")
    train.class_is_last()
    # build classifier
    cls = Classifier(classname="weka.classifiers.lazy.IBk")
    cls.build_classifier(train)
    print(cls)

    # evaluate
    evl = Evaluation(train)
    evl.test_model(cls, test)
    print(evl.summary())
    print(evl.class_details())
    print(evl.matrix())

    jvm.stop()
   
print "              "   
print "Computing Done"



