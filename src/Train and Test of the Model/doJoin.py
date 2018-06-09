#IMPORTS
import sys
import os
import csv
import numpy
from sklearn import preprocessing

#INFO
print ("--------------------------------------------------------------------Info-----------------------------------------------------------------")
print ("This is a script that unifies separated genre statistics into a single .csv file with all the genres")
print ("                                                                                                                                         ")
print ("David Campayo, Final Degree Project, Audiovisual Systems Engineering, 2018                                                               ")
print ("-----------------------------------------------------------------------------------------------------------------------------------------")
print ("                                                                                                                                         ")

#READING COMMAND LINE

if len(sys.argv) < 3:
    print ("----------------------------- How to use it-------------------------------")
    print ("                                                                 ")
    print ("Declaration:            -> doJoin_Convenconal.py path\\to\\data\\folder path\\to\\data\\saving")

    sys.exit(-1)


#DECLARATION DATA
statisticsDataFolder = sys.argv[1]
classifierDataSaving = sys.argv[2]

statistics_allgenres = []
genre = []
descriptors = []

#----------------------------------------Opening Statistics from all genres and unifiying into a single matrix-------------------------------------
with open(statisticsDataFolder+ "\\Statistics_blues.csv", "r") as statisticsblu:
    statisticsblu.readline()
    statisticsblu = csv.reader(statisticsblu)
    for row in statisticsblu:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])

with open(statisticsDataFolder+ "\\Statistics_classical.csv", "r") as statisticscla:
    statisticscla.readline()
    statisticscla = csv.reader(statisticscla)
    for row in statisticscla:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])

with open(statisticsDataFolder+ "\\Statistics_country.csv", "r") as statisticscoun:
    statisticscoun.readline()
    statisticscoun = csv.reader(statisticscoun)
    for row in statisticscoun:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])
    
with open(statisticsDataFolder+ "\\Statistics_disco.csv", "r") as statisticsdis:
    statisticsdis.readline()
    statisticsdis = csv.reader(statisticsdis)
    for row in statisticsdis:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])
    
with open(statisticsDataFolder+ "\\Statistics_hiphop.csv", "r") as statisticship:
    statisticship.readline()
    statisticship = csv.reader(statisticship)
    for row in statisticship:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])

with open(statisticsDataFolder+ "\\Statistics_jazz.csv", "r") as statisticsja:
    statisticsja.readline()
    statisticsja = csv.reader(statisticsja)
    for row in statisticsja:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])

with open(statisticsDataFolder+ "\\Statistics_metal.csv", "r") as statisticsme:
    statisticsme.readline()
    statisticsme = csv.reader(statisticsme)
    for row in statisticsme:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])

with open(statisticsDataFolder+ "\\Statistics_pop.csv", "r") as statisticspo:
    statisticspo.readline()
    statisticspo = csv.reader(statisticspo)
    for row in statisticspo:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])
        
with open(statisticsDataFolder+ "\\Statistics_reggae.csv", "r") as statisticsreg:
    statisticsreg.readline()
    statisticsreg = csv.reader(statisticsreg)
    for row in statisticsreg:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])
        
with open(statisticsDataFolder+ "\\Statistics_rock.csv", "r") as statisticsro:
    statisticsro.readline()
    statisticsro = csv.reader(statisticsro)
    for row in statisticsro:
        statistics_allgenres.append(row[0:len(row)-1])
        genre.append(row[len(row)-1])


statistics_allgenres = preprocessing.scale(statistics_allgenres)

write_file = "classifier_allgenres.csv"
with open(classifierDataSaving +"\\"+write_file, "w") as csvfile:
    csv_classifier_allgenres = csv.writer(csvfile)
    for row in statistics_allgenres:
        csv_classifier_allgenres.writerows([row])
statis=[]
with open("C:\Users\David\Desktop\TFG_2\\3_AllClassifiers\Test\classifier_allgenres.csv","r") as csvfile:
    csv_classifiers_allgenres = csv.reader(csvfile)
    csvfile = csv.reader(csvfile)
    for row in csvfile:
        statis.append(row)
    i = 0
    for row in statis:
        row.append(genre[i])
        i = i +1

descriptors = ["MeanLogattackTime","MeanRms","MeanSpectralCentroid","MeanSpectralCrest","MeanSpectralFlatness","MeanSpectralFlux","MeanSpectralKurtosis","MeanSpectralRollOff","MeanSpectralSkewness","MeanSpectralSpread","MeanTemporalCentroid","MeanZeroCrossingRate","MeanMfcc0","MeanMfcc1","MeanMfcc2","MeanMfcc3","MeanMfcc4","MeanMfcc5","MeanMfcc6","MeanMfcc7","MeanMfcc8","MeanMfcc9","MeanMfcc10","MeanMfcc11","MeanMfcc12","MeanMfcc13","MeanMfcc14","MeanMfcc15","MeanMfcc16","MeanMfcc17","MeanMfcc18","MeanMfcc19","Genre"]       
write_file = "classifier_allgenres.csv"
with open(classifierDataSaving +"\\"+write_file, "w") as csvfile:
    csv_classifier_allgenres = csv.writer(csvfile)
    csv_classifier_allgenres.writerow(descriptors)
    for row in statis:
        csv_classifier_allgenres.writerows([row])    
        

print "Computing Done"




