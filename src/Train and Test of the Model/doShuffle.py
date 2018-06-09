#IMPORTS
import sys
import os
import csv
import numpy
import random
import scipy

#INFO
print ("--------------------------------------------------------------------Info-----------------------------------------------------------------")
print ("This is a script that load all genre statistics joined and shuffles the data.                                                            ")
print ("                                                                                                                                         ")
print ("David Campayo, Final Degree Project, Audiovisual Systems Engineering, 2018                                                               ")
print ("-----------------------------------------------------------------------------------------------------------------------------------------")
print ("                                                                                                                                         ")

#READING COMMAND LINE

if len(sys.argv) < 3:
    print ("----------------------------- How to use it-------------------------------")
    print ("                                                                 ")
    print ("Declaration:            -> doShuffle.py path\\to\\data\\folder path\\to\\data\\saving")

    sys.exit(-1)

#DECLARATION DATA
statisticsAllGenresDataFolder = sys.argv[1]
statisticsAllGenresShuffleDataSaving = sys.argv[2]

all_Genres = []
headers = []
all_Genres_Shuffled = []

#1. Searching Files

statisticsAllGenresList = os.listdir(statisticsAllGenresDataFolder)
print ("This is your statistics database:  ")
print ("                                   ")
print statisticsAllGenresList

for statisticFile in statisticsAllGenresList:
    statisticsAllGenresFilePath = (os.path.join(statisticsAllGenresDataFolder,statisticFile))
    print (statisticsAllGenresFilePath)
            
#----------------------------------Opening Eigen Vectors Matrix from all genres-----------------------------
with open("C:\Users\David\Desktop\TFG_2\\3_AllClassifiers\classifier_allgenres.csv", "r") as AllGenres:
    AllGenres = csv.reader(AllGenres)
    for row in AllGenres:
       all_Genres.append(row)
    headers = all_Genres[0]
    data = all_Genres[1:len(all_Genres)]
numpy.random.shuffle(data)
all_Genres_Shuffled.append(headers)
all_Genres_Shuffled.extend(data)

write_file = "allgenres_shuffled.csv"
with open(statisticsAllGenresShuffleDataSaving +"\\"+write_file, "w") as csvfile:
    allGenresShuffled = csv.writer(csvfile)
    for row in all_Genres_Shuffled:
        allGenresShuffled.writerows([row])                                  
print "Computing Done"



