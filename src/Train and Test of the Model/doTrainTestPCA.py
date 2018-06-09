#IMPORTS
import sys
import os
import csv
import numpy
import random
from sklearn import preprocessing

#INFO
print ("--------------------------------------------------------------------Info-----------------------------------------------------------------")
print ("This is a script that makes train and test data for each fold and PCA genre model and saves it in a .csv file                ")
print ("                                                                                                                                         ")
print ("David Campayo, Final Degree Project, Audiovisual Systems Engineering, 2018                                                               ")
print ("-----------------------------------------------------------------------------------------------------------------------------------------")
print ("                                                                                                                                         ")

#READING COMMAND LINE

if len(sys.argv) < 3:
    print ("----------------------------- How to use it-------------------------------")
    print ("                                                                 ")
    print ("Declaration:            -> doTrainTestPCA.py path\\to\\data\\folder\\EigenVectors path\\to\\data\\folder\\All Genres Folds Shuffled path\\to\\data\\saving")

    sys.exit(-1)

#DECLARATION DATA
eigenvectorsDataFolder = sys.argv[1]
statisticsAllGenresShuffledFoldsDataFolder= sys.argv[2]
statisticsAllGenresShuffledFoldsPCADataSaving = sys.argv[3]

eigen_Vectors = []
headers = []
data = []
#test_PCA_genre = []

#EigenVectors
eigenVectorsBlues, eigenVectorsClassical, eigenVectorsCountry, eigenVectorsDisco, eigenVectorsHipHop, eigenVectorsJazz, eigenVectorsMetal, eigenVectorsPop, eigenVectorsReggae, eigenVectorsRock = ([] for x in range(10))
listofEigenVectors = [eigenVectorsBlues, eigenVectorsClassical, eigenVectorsCountry, eigenVectorsDisco, eigenVectorsHipHop, eigenVectorsJazz, eigenVectorsMetal, eigenVectorsPop, eigenVectorsReggae, eigenVectorsRock]

#Test 
test_fold0, test_fold1, test_fold2, test_fold3, test_fold4, test_fold5, test_fold6, test_fold7, test_fold8, test_fold9 = ([] for i in range(10))
listofTestFolds = [test_fold0, test_fold1, test_fold2, test_fold3, test_fold4, test_fold5, test_fold6, test_fold7, test_fold8, test_fold9]

genre_test_fold0, genre_test_fold1, genre_test_fold2, genre_test_fold3, genre_test_fold4, genre_test_fold5, genre_test_fold6, genre_test_fold7, genre_test_fold8, genre_test_fold9 = ([] for i in range(10))
listofTestFoldsGenre = [genre_test_fold0, genre_test_fold1, genre_test_fold2, genre_test_fold3, genre_test_fold4, genre_test_fold5, genre_test_fold6, genre_test_fold7, genre_test_fold8, genre_test_fold9]

#Train
train_fold0, train_fold1, train_fold_2, train_fold3, train_fold4, train_fold5, train_fold6, train_fold7, train_fold8, train_fold9 = ([] for i in range(10))
listofTrainFolds = [train_fold0, train_fold1, train_fold_2, train_fold3, train_fold4, train_fold5, train_fold6, train_fold7, train_fold8, train_fold9]

genre_train_fold0, genre_train_fold1, genre_train_fold2, genre_train_fold3, genre_train_fold4, genre_train_fold5, genre_train_fold6, genre_train_fold7, genre_train_fold8, genre_train_fold9 = ([] for i in range(10))
listofTrainFoldsGenre = [genre_train_fold0, genre_train_fold1, genre_train_fold2, genre_train_fold3, genre_train_fold4, genre_train_fold5, genre_train_fold6, genre_train_fold7,genre_train_fold8, genre_train_fold9]


#1. Searching Files
eigenvectorsDataFolderList = os.listdir(eigenvectorsDataFolder)
print ("These are your genre eigenvectors database:       ")
print ("                                                  ")
print eigenvectorsDataFolderList
print ("                                                  ")

statisticsAllGenresShuffledFoldsDataFolderList = os.listdir(statisticsAllGenresShuffledFoldsDataFolder)
print ("These are all genres shuffled folds database:     ")
print ("                                                  ")
print statisticsAllGenresShuffledFoldsDataFolderList
print ("                                                  ")

print ("These are EigenVectors file paths:                ")
print ("                                                  ")

for statisticFileEigenVectors in eigenvectorsDataFolderList:
    eigenvectorsDataFolderFilePath = (os.path.join(eigenvectorsDataFolder,statisticFileEigenVectors))
    print (eigenvectorsDataFolderFilePath)
print len(eigenvectorsDataFolderList)
print ("                                                  ")
print ("These are AlLGenresFolds file paths:              ")
print ("                                                  ")
for statisticFileAllGenresFolds in statisticsAllGenresShuffledFoldsDataFolderList:
    statisticsAllGenresShuffledFoldsFilePath = (os.path.join(statisticsAllGenresShuffledFoldsDataFolder,statisticFileAllGenresFolds))
    print (statisticsAllGenresShuffledFoldsFilePath)
print ("                                              ")
print len(statisticsAllGenresShuffledFoldsDataFolderList)
            
#------------------------------------Opening All Genres Eigenvectors ---------------------------------
print ("Computing Start")
print ("               ")

for statisticFileEigenVectors in eigenvectorsDataFolderList:
    eigenvectorsDataFolderFilePath = (os.path.join(eigenvectorsDataFolder,statisticFileEigenVectors))
    print(eigenvectorsDataFolderFilePath)
    with open(eigenvectorsDataFolderFilePath, "r") as eigenVectors:
        eigenVectors.readline()
        eigenVectors = csv.reader(eigenVectors)
        if eigenvectorsDataFolderFilePath.endswith("_blues.csv"):
            for row in eigenVectors:
                eigenVectorsBlues.append(row[0:len(row)-1])
        if eigenvectorsDataFolderFilePath.endswith("_classical.csv"):
            for row in eigenVectors:
                eigenVectorsClassical.append(row[0:len(row)-1])
        if eigenvectorsDataFolderFilePath.endswith("_country.csv"):
            for row in eigenVectors:
                eigenVectorsCountry.append(row[0:len(row)-1])
        if eigenvectorsDataFolderFilePath.endswith("_disco.csv"):
            for row in eigenVectors:
                eigenVectorsDisco.append(row[0:len(row)-1])
        if eigenvectorsDataFolderFilePath.endswith("_hiphop.csv"):
            for row in eigenVectors:
                eigenVectorsHipHop.append(row[0:len(row)-1])
        if eigenvectorsDataFolderFilePath.endswith("_jazz.csv"):
            for row in eigenVectors:
                eigenVectorsJazz.append(row[0:len(row)-1])
        if eigenvectorsDataFolderFilePath.endswith("_metal.csv"):
            for row in eigenVectors:
                eigenVectorsMetal.append(row[0:len(row)-1])
        if eigenvectorsDataFolderFilePath.endswith("_pop.csv"):
            for row in eigenVectors:
                eigenVectorsPop.append(row[0:len(row)-1])
        if eigenvectorsDataFolderFilePath.endswith("_reggae.csv"):
            for row in eigenVectors:
                eigenVectorsReggae.append(row[0:len(row)-1])
        if eigenvectorsDataFolderFilePath.endswith("_rock.csv"):
            for row in eigenVectors:
                eigenVectorsRock.append(row[0:len(row)-1])

#print listofEigenVectors

#------------------------------------Opening all test folds ---------------------------------
                
genreFoldsTest=[]
for statisticFileAllGenresFolds in statisticsAllGenresShuffledFoldsDataFolderList:
    statisticsAllGenresShuffledFoldsFilePath = (os.path.join(statisticsAllGenresShuffledFoldsDataFolder,statisticFileAllGenresFolds))
    with open(statisticsAllGenresShuffledFoldsFilePath, "r") as allGenresShuffledFolds:
        #allGenresShuffledFolds.readline()
        allGenresShuffledFolds = csv.reader(allGenresShuffledFolds)
        n = 0
        for itemTest in range(0,len(listofTestFolds)):
            if statisticsAllGenresShuffledFoldsFilePath.endswith(str(n)+"_test.csv"):
                for row in allGenresShuffledFolds:
                    listofTestFolds[itemTest].append(row[0:len(row)-1]) #Saving the values of descriptors
                    for itemTestGenre in range(0, len(listofTestFoldsGenre)):
                        listofTestFoldsGenre[itemTestGenre].append(row[32]) #Saving genre of the song
                genreFoldsTest.append(listofTestFoldsGenre[itemTestGenre])
                listofTestFoldsGenre[itemTestGenre] = []
            n = n+1
#------------------------------------Opening all train folds---------------------------------

genreFoldsTrain=[]
for statisticFileAllGenresFolds in statisticsAllGenresShuffledFoldsDataFolderList:
    statisticsAllGenresShuffledFoldsFilePath = (os.path.join(statisticsAllGenresShuffledFoldsDataFolder,statisticFileAllGenresFolds))
    with open(statisticsAllGenresShuffledFoldsFilePath, "r") as allGenresShuffledFolds:
        #allGenresShuffledFolds.readline()
        allGenresShuffledFolds = csv.reader(allGenresShuffledFolds)
        m = 0
        for itemTrain in range(0,len(listofTrainFolds)):
            if statisticsAllGenresShuffledFoldsFilePath.endswith(str(m)+"_train.csv"):
                for row in allGenresShuffledFolds:
                    listofTrainFolds[itemTrain].append(row[0:len(row)-1]) #Saving the values of descriptors
                    for itemTrainGenre in range(0, len(listofTrainFoldsGenre)):
                        listofTrainFoldsGenre[itemTrainGenre].append(row[32]) #Saving genre of the song
                genreFoldsTrain.append(listofTrainFoldsGenre[itemTrainGenre])
                listofTrainFoldsGenre[itemTrainGenre] = []
            m = m+1

#------------------------------------Multiplying each test fold by each genre eigenvector (PCA Genre)---------------------------------

genre = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
test_PCA_genre = []
test_PCA_genre_nogenre = []

for itemTest in range(0,len(listofTestFolds)):
    numpy.float32(listofTestFolds[itemTest]) 
    for itemEigen in range(0, len(listofEigenVectors)):
        numpy.float32(listofEigenVectors[itemEigen])
        #test_PCA_genre.append(numpy.dot(numpy.float32(listofTestFolds[itemTest]),numpy.float32(listofEigenVectors[itemEigen])))
        test_PCA_genre.append([[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*numpy.float32(listofEigenVectors[itemEigen]))] for X_row in numpy.float32(listofTestFolds[itemTest])])
        if itemTest==0:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        if itemTest==1:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        if itemTest==2:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        if itemTest==3:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        if itemTest==4:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        if itemTest==5:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        if itemTest==6:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        if itemTest==7:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        if itemTest==8:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        if itemTest==9:
            test_PCA_genre_nogenre.append(genreFoldsTest[itemTest])
        print "ITEM TEST"
        print itemTest
        print "ITEM EIGEN"
        print itemEigen

it = []
ite = []
gen = ["blu", "cla", "count", "disco", "hip","jazz","met","pop","reg","rock"] 
nogen = ["no blu", "no cla", "no count", "no disco", "no hip", "no jazz", "no met","no pop","no reg","no rock"]

for i in range(0,10):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[i]:
                it = [x.replace(item,nogen[i]) for x in it]
            ite.extend(it)
            it = []

for i in range(10,20):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-10)]:
                it = [x.replace(item,nogen[(i-10)]) for x in it]
            ite.extend(it)
            it = []
for i in range(20,30):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-20)]:
                it = [x.replace(item,nogen[(i-20)]) for x in it]
            ite.extend(it)
            it = []
for i in range(30,40):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-30)]:
                it = [x.replace(item,nogen[(i-30)]) for x in it]
            ite.extend(it)
            it = []
for i in range(40,50):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-40)]:
                it = [x.replace(item,nogen[(i-40)]) for x in it]
            ite.extend(it)
            it = []
for i in range(50,60):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-50)]:
                it = [x.replace(item,nogen[(i-50)]) for x in it]
            ite.extend(it)
            it = []
for i in range(60,70):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-70)]:
                it = [x.replace(item,nogen[(i-70)]) for x in it]
            ite.extend(it)
            it = []
for i in range(70,80):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-70)]:
                it = [x.replace(item,nogen[(i-70)]) for x in it]
            ite.extend(it)
            it = []
for i in range(80,90):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-80)]:
                it = [x.replace(item,nogen[(i-80)]) for x in it]
            ite.extend(it)
            it = []
for i in range(90,100):
    print i
    for item in test_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-90)]:
                it = [x.replace(item,nogen[(i-90)]) for x in it]
            ite.extend(it)
            it = []
#---------------------------------------------------------------------------------
i = 0
for row in test_PCA_genre:
    for item in row:
        item.append(ite[i])
        i = i+1
        print i
#---------------------------------------------------------------------------------
print len(test_PCA_genre_nogenre[0])

itest = 0
for itemTest in range(0,len(listofTestFolds)):
    for itemGenre in range(0, len(genre)):
        write_file = "data_fold"+str(itemTest)+"_test_PCA_"+str(genre[itemGenre])+".csv"
        with open(statisticsAllGenresShuffledFoldsPCADataSaving +"\\"+write_file, "w") as csvfile:
            allGenresPCA_test = csv.writer(csvfile)
            if genre[itemGenre] == "blues":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","Genre"])
            if genre[itemGenre] == "classical":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","Genre"])
            if genre[itemGenre] == "country":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","Genre"])
            if genre[itemGenre] == "disco":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","Genre"])
            if genre[itemGenre] == "hiphop":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","Genre"])
            if genre[itemGenre] == "jazz":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","Genre"])
            if genre[itemGenre] == "metal":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","Genre"])
            if genre[itemGenre] == "pop":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","Genre"])
            if genre[itemGenre] == "reggae":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","Genre"])
            if genre[itemGenre] == "rock":
                allGenresPCA_test.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","Genre"])
                      
            #allGenresPCA_test.writerow(["PC1, PC2, PC3, PC4"])
            allGenresPCA_test.writerows(test_PCA_genre[itest])
            itest = itest +1
            print itest
write_file = "test_PCA_genre_nogenre.csv"
with open(statisticsAllGenresShuffledFoldsPCADataSaving +"\\"+write_file, "w") as csvfile:
    alltest_PCA_genre_nogenre = csv.writer(csvfile)
    alltest_PCA_genre_nogenre.writerows(ite)
   
#------------------------------------Multiplying each train fold by each genre eigenvector (PCA Genre)---------------------------------
ite2 = []
train_PCA_genre = []
train_PCA_genre_nogenre = []

for itemTrain in range(0,len(listofTrainFolds)):
    numpy.float32(listofTrainFolds[itemTrain])
    for itemEigen in range(0, len(listofEigenVectors)):
        numpy.float32(listofEigenVectors[itemEigen])
        train_PCA_genre.append([[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*numpy.float32(listofEigenVectors[itemEigen]))] for X_row in numpy.float32(listofTrainFolds[itemTrain])])
        #train_PCA_genre.append(numpy.dot(numpy.float32(listofTrainFolds[itemTrain]),numpy.float32(listofEigenVectors[itemEigen])))
        print len(train_PCA_genre)
        if itemTrain==0:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        if itemTrain==1:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        if itemTrain==2:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        if itemTrain==3:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        if itemTrain==4:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        if itemTrain==5:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        if itemTrain==6:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        if itemTrain==7:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        if itemTrain==8:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        if itemTrain==9:
            train_PCA_genre_nogenre.append(genreFoldsTrain[itemTrain])
        print "ITEM TRAIN"
        print itemTrain
        print "ITEM EIGEN"
        print itemEigen
print train_PCA_genre_nogenre
print len(train_PCA_genre_nogenre)

for i in range(0,10):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[i]:
                it = [x.replace(item,nogen[i]) for x in it]
            ite2.extend(it)
            it = []

for i in range(10,20):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-10)]:
                it = [x.replace(item,nogen[(i-10)]) for x in it]
            ite2.extend(it)
            it = []
for i in range(20,30):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-20)]:
                it = [x.replace(item,nogen[(i-20)]) for x in it]
            ite2.extend(it)
            it = []
for i in range(30,40):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-30)]:
                it = [x.replace(item,nogen[(i-30)]) for x in it]
            ite2.extend(it)
            it = []
for i in range(40,50):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-40)]:
                it = [x.replace(item,nogen[(i-40)]) for x in it]
            ite2.extend(it)
            it = []
for i in range(50,60):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-50)]:
                it = [x.replace(item,nogen[(i-50)]) for x in it]
            ite2.extend(it)
            it = []
for i in range(60,70):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-70)]:
                it = [x.replace(item,nogen[(i-70)]) for x in it]
            ite2.extend(it)
            it = []
for i in range(70,80):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-70)]:
                it = [x.replace(item,nogen[(i-70)]) for x in it]
            ite2.extend(it)
            it = []
for i in range(80,90):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-80)]:
                it = [x.replace(item,nogen[(i-80)]) for x in it]
            ite2.extend(it)
            it = []
for i in range(90,100):
    print i
    for item in train_PCA_genre_nogenre[i]:
        it.append(item)
        for item in it:
            if item != gen[(i-90)]:
                it = [x.replace(item,nogen[(i-90)]) for x in it]
            ite2.extend(it)
            it = []

print len (ite2)
#---------------------------------------------------------------------------------
i = 0
for row in train_PCA_genre:
    for item in row:
        item.append(ite2[i])
        i = i+1
        print i
#---------------------------------------------------------------------------------

itrain = 0
for itemTrain in range(0,len(listofTrainFolds)):
    for itemGenre in range(0, len(genre)):
        write_file = "data_fold"+str(itemTrain)+"_train_PCA_"+str(genre[itemGenre])+".csv"
        with open(statisticsAllGenresShuffledFoldsPCADataSaving +"\\"+write_file, "w") as csvfile:
            allGenresPCA_train = csv.writer(csvfile)
            if genre[itemGenre] == "blues":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","Genre"])
            if genre[itemGenre] == "classical":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","Genre"])
            if genre[itemGenre] == "country":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","Genre"])
            if genre[itemGenre] == "disco":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","Genre"])
            if genre[itemGenre] == "hiphop":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","Genre"])
            if genre[itemGenre] == "jazz":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","Genre"])
            if genre[itemGenre] == "metal":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","Genre"])
            if genre[itemGenre] == "pop":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","Genre"])
            if genre[itemGenre] == "reggae":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20","Genre"])
            if genre[itemGenre] == "rock":
                allGenresPCA_train.writerow(["PC1", "PC2", "PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16","PC17","PC18","PC19","Genre"])
            #allGenresPCA_test.writerow(["PC1, PC2, PC3, PC4"])
            allGenresPCA_train.writerows(train_PCA_genre[itrain])
            itrain = itrain +1
            print itrain
write_file = "train_PCA_genre_nogenre.csv"
with open(statisticsAllGenresShuffledFoldsPCADataSaving +"\\"+write_file, "w") as csvfile:
    alltrain_PCA_genre_nogenre = csv.writer(csvfile)
    alltrain_PCA_genre_nogenre.writerows(ite2)
            
print "              "   
print "Computing Done"



