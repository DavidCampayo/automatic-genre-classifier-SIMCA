#IMPORTS
import sys
import os
import csv
import numpy
import random

#INFO
print ("--------------------------------------------------------------------Info-----------------------------------------------------------------")
print ("This is a script that makes K fold and saves train and test .csv file in each fold.                                    ")
print ("                                                                                                                                         ")
print ("David Campayo, Final Degree Project, Audiovisual Systems Engineering, 2018                                                               ")
print ("-----------------------------------------------------------------------------------------------------------------------------------------")
print ("                                                                                                                                         ")

#READING COMMAND LINE

if len(sys.argv) < 3:
    print ("----------------------------- How to use it-------------------------------")
    print ("                                                                 ")
    print ("Declaration:            -> doKfold_David.py path\\to\\data\\folder path\\to\\data\\saving")

    sys.exit(-1)

#DECLARATION DATA
statisticsAllGenresShuffledDataFolder = sys.argv[1]
statisticsTrainTestTenFoldDataSaving = sys.argv[2]
all_Genres = []

#1. Searching Files

statisticsAllGenresShuffledList = os.listdir(statisticsAllGenresShuffledDataFolder)
print ("This is your all genres shuffled database:  ")
print ("                                            ")
print statisticsAllGenresShuffledList
print ("                                            ")

for statisticFile in statisticsAllGenresShuffledList:
    statisticsAllGenresShuffledFilePath = (os.path.join(statisticsAllGenresShuffledDataFolder,statisticFile))
    print ("This is the file path:                  ")
    print ("                                        ")
    print (statisticsAllGenresShuffledFilePath)
    print ("                                        ")
            
#------------------------------------Opening All Genres Shuffled data ---------------------------------
print ("Computing Start")
print ("               ")
with open("C:\Users\David\Desktop\TFG_2\\4_AllClassifiers_Shuffled\\allgenres_shuffled.csv", "r") as AllGenres:
    AllGenres = csv.reader(AllGenres)
    for row in AllGenres:
       all_Genres.append(row)
    headers = all_Genres[0]
    data = all_Genres[1:len(all_Genres)]
#------------------------------------------Initializing variables---------------------------------------
Kfolds = 10 #Change K to select a proper number of folds
first_test = 0
last_test = (len(data)/Kfolds)
#----------------------------Computing train and test .csv file for each fold--------------------------
print ("Folds selected: "+str(Kfolds))
print ("                             ")
for fold in range(0,Kfolds):
    print "Computing train and test .csv file for fold: "+str(fold)
#-------------------------------------Computing test file----------------------------------------------
    test_data = data[first_test:last_test]
    write_file = "data_fold"+str(fold)+"_test.csv"
    with open(statisticsTrainTestTenFoldDataSaving +"\\"+write_file, "w") as csvfile:
        allGenres10fold_test = csv.writer(csvfile)
        for row in test_data:
            allGenres10fold_test.writerows([row])
#-------------------------------------Computing train file----------------------------------------------
    train_data = numpy.delete(data,slice(first_test,last_test), axis = 0)
    write_file = "data_fold"+str(fold)+"_train.csv"
    with open(statisticsTrainTestTenFoldDataSaving +"\\"+write_file, "w") as csvfile:
        allGenres10fold_train = csv.writer(csvfile)
        for row in train_data:
            allGenres10fold_train.writerows([row])
#-----------------------------------Iterating over all combinations-------------------------------------           
    first_test = last_test
    last_test = first_test + ((len(data)/Kfolds))
    
print "              "   
print "Computing Done"



