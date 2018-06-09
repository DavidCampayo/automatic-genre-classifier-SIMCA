#IMPORTS
import sys
import os
import csv
import itertools
import numpy
import scipy
from javabridge import JWrapper, JClassWrapper
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from weka.filters import Filter
import weka.core.jvm as jvm


jvm.start()

#INFO
print ("--------------------------------------------------------------------Info-----------------------------------------------------------------")
print ("This is a script that load genre statistics data performed by doStatistics_David.py to Weka,filter the data using Principal Component")
print ("                                                                                                                                         ")
print ("David Campayo, Final Degree Project, Audiovisual Systems Engineering, 2018                                                               ")
print ("-----------------------------------------------------------------------------------------------------------------------------------------")
print ("                                                                                                                                         ")

#READING COMMAND LINE

if len(sys.argv) < 3:
    print ("----------------------------- How to use it-------------------------------")
    print ("                                                                 ")
    print ("Declaration:            -> doSingleClassifiers.py path\\to\\data\\folder path\\to\\data\\saving")

    sys.exit(-1)


#DECLARATION DATA
statisticsDataFolder = sys.argv[1]
singleClassifierDataSaving = sys.argv[2]

eigen_blu = []
eigen_cla = []
eigen_coun = []
eigen_dis = []
eigen_hip = []
eigen_ja = []
eigen_me = []
eigen_po = []
eigen_reg = []
eigen_ro= []


eigenvectors_blues = []
eigenvectors_classical = []
eigenvectors_country = []
eigenvectors_disco = []
eigenvectors_hiphop = []
eigenvectors_jazz = []
eigenvectors_metal = []
eigenvectors_pop = []
eigenvectors_reggae = []
eigenvectors_rock = []


statistics_blu = []
statistics_cla = []
statistics_coun = []
statistics_dis = []
statistics_hip = []
statistics_ja = []
statistics_me = []
statistics_po = []
statistics_reg = []
statistics_ro = []


statistics_blues = []
statistics_classical = []
statistics_country = []
statistics_disco = []
statistics_hiphop = []
statistics_jazz = []
statistics_metal = []
statistics_pop = []
statistics_reggae = []
statistics_rock = []


data_blues_no_blues = []
data_classical_no_classical = []
data_country_no_country = []
data_disco_no_disco = []
data_hiphop_no_hiphop = []
data_jazz_no_jazz = []
data_metal_no_metal = []
data_pop_no_pop = []
data_reggae_no_reggae = []
data_rock_no_rock = []


single_classifier_blues_no_blues = []
single_classifier_classical_no_classical = []
single_classifier_country_no_country = []
single_classifier_disco_no_disco = []
single_classifier_hiphop_no_hiphop = []
single_classifier_jazz_no_jazz = []
single_classifier_metal_no_metal = []
single_classifier_pop_no_pop = []
single_classifier_reggae_no_reggae = []
single_classifier_rock_no_rock = []

#1. Searching Files

statisticsList = os.listdir(statisticsDataFolder)
print ("This is your statistics database:  ")
print ("                                   ")
print statisticsList

for statisticFile in statisticsList:
    statisticFilePath = (os.path.join(statisticsDataFolder,statisticFile))
    print (statisticFilePath)
            
#----------------------------------Opening Eigen Vectors Matrix from all genres-----------------------------
with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_blues.csv", "r") as eigenblues:
    eigenblues.readline()
    eigenblues = csv.reader(eigenblues)
    for row in eigenblues:
        last_attribute = len(row)-1
        row = map(float,row[0:last_attribute])
        eigen_blu.append(row)
    #eigenvectors_blues.append(eigen_blu[0])
    eigenvectors_blues.extend(eigen_blu[0:32])
    print eigenvectors_blues

##with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_classical.csv", "r") as eigenclassical:
##    eigenclassical.readline()
##    eigenclassical = csv.reader(eigenclassical)
##    for row in eigenclassical:
##        last_attribute = len(row)-1
##        row = map(float,row[0:last_attribute])
##        eigen_cla.append(row)
##    #eigenvectors_classical.append(eigen_cla[0])
##    eigenvectors_classical.extend(eigen_cla[0:32])

with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_country.csv", "r") as eigencountry:
    eigencountry.readline()
    eigencountry = csv.reader(eigencountry)
    for row in eigencountry:
        last_attribute = len(row)-1
        row = map(float,row[0:last_attribute])
        eigen_coun.append(row)
    #eigenvectors_country.append(eigen_coun[0])
    eigenvectors_country.extend(eigen_coun[0:32])

##with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_disco.csv", "r") as eigendisco:
##    eigendisco.readline()
##    eigendisco = csv.reader(eigendisco)
##    for row in eigendisco:
##        last_attribute = len(row)-1
##        row = map(float,row[0:last_attribute])
##        eigen_dis.append(row)
##    #eigenvectors_disco.append(eigen_dis[0])
##    eigenvectors_disco.extend(eigen_dis[0:32])

with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_hiphop.csv", "r") as eigenhiphop:
    eigenhiphop.readline()
    eigenhiphop = csv.reader(eigenhiphop)
    for row in eigenhiphop:
        last_attribute = len(row)-1
        row = map(float,row[0:last_attribute])
        eigen_hip.append(row)
    #eigenvectors_hiphop.append(eigen_hip[0])
    eigenvectors_hiphop.extend(eigen_hip[0:32])

with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_jazz.csv", "r") as eigenjazz:
    eigenjazz.readline()
    eigenjazz = csv.reader(eigenjazz)
    for row in eigenjazz:
        last_attribute = len(row)-1
        row = map(float,row[0:last_attribute])
        eigen_ja.append(row)
    #eigenvectors_jazz.append(eigen_ja[0])
    eigenvectors_jazz.extend(eigen_ja[0:32])

##with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_metal.csv", "r") as eigenmetal:
##    eigenmetal.readline()
##    eigenmetal = csv.reader(eigenmetal)
##    for row in eigenmetal:
##        last_attribute = len(row)-1
##        row = map(float,row[0:last_attribute])
##        eigen_me.append(row)
##    #eigenvectors_metal.append(eigen_me[0])
##    eigenvectors_metal.extend(eigen_me[0:32])

with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_pop.csv", "r") as eigenpop:
    eigenpop.readline()
    eigenpop = csv.reader(eigenpop)
    for row in eigenpop:
        last_attribute = len(row)-1
        row = map(float,row[0:last_attribute])
        eigen_po.append(row)
    #eigenvectors_pop.append(eigen_po[0])
    eigenvectors_pop.extend(eigen_po[0:32])

##with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_reggae.csv", "r") as eigenreggae:
##    eigenreggae.readline()
##    eigenreggae = csv.reader(eigenreggae)
##    for row in eigenreggae:
##        last_attribute = len(row)-1
##        row = map(float,row[0:last_attribute])
##        eigen_reg.append(row)
##    #eigenvectors_reggae.append(eigen_reg[0])
##    eigenvectors_reggae.extend(eigen_reg[0:32])

with open("C:\Users\David\Desktop\TFG\\7_EigenVectors\EigenVectors_rock.csv", "r") as eigenrock:
    eigenrock.readline()
    eigenrock = csv.reader(eigenrock)
    for row in eigenrock:
        last_attribute = len(row)-1
        row = map(float,row[0:last_attribute])
        eigen_ro.append(row)
    #eigenvectors_rock.append(eigen_ro[0])
    eigenvectors_rock.extend(eigen_ro[0:32])


#----------------------------------------Opening Statistics from all genres--------------------------------------------
#with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_blues.csv", "r") as statisticsblu:
with open("C:\Users\David\Desktop\TFG\\10_Train_Test_Externa\\5_Descriptors_Statistics_Data_Test\Statistics_blues.csv", "r") as statisticsblu:
    statisticsblu.readline()
    statisticsblu = csv.reader(statisticsblu)
    for row in statisticsblu:
        row = map(float,row[0:32])
        statistics_blu.append(row[0:32])
    statistics_blues.append(statistics_blu[0])
    statistics_blues.extend(statistics_blu[1:101])
    print statistics_blues
    statistics_blues = preprocessing.scale(statistics_blues)

##with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_classical.csv", "r") as statisticscla:
##    statisticscla.readline()
##    statisticscla = csv.reader(statisticscla)
##    for row in statisticscla:
##        row = map(float,row[0:32])
##        statistics_cla.append(row[0:32])
##    statistics_classical.append(statistics_cla[0])
##    statistics_classical.extend(statistics_cla[1:101])
##    statistics_classical = preprocessing.scale(statistics_classical)

#with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_country.csv", "r") as statisticscoun:
with open("C:\Users\David\Desktop\TFG\\10_Train_Test_Externa\\5_Descriptors_Statistics_Data_Test\Statistics_country.csv", "r") as statisticscoun:
    statisticscoun.readline()
    statisticscoun = csv.reader(statisticscoun)
    for row in statisticscoun:
        row = map(float,row[0:32])
        statistics_coun.append(row[0:32])
    statistics_country.append(statistics_coun[0])
    statistics_country.extend(statistics_coun[1:101])
    statistics_country = preprocessing.scale(statistics_country)
    
##with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_disco.csv", "r") as statisticsdis:
##    statisticsdis.readline()
##    statisticsdis = csv.reader(statisticsdis)
##    for row in statisticsdis:
##        row = map(float,row[0:32])
##        statistics_dis.append(row[0:32])
##    statistics_disco.append(statistics_dis[0])
##    statistics_disco.extend(statistics_dis[1:101])
##    statistics_disco = preprocessing.scale(statistics_disco)
    
#with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_hiphop.csv", "r") as statisticship:
with open("C:\Users\David\Desktop\TFG\\10_Train_Test_Externa\\5_Descriptors_Statistics_Data_Test\Statistics_hiphop.csv", "r") as statisticship:
    statisticship.readline()
    statisticship = csv.reader(statisticship)
    for row in statisticship:
        row = map(float,row[0:32])
        statistics_hip.append(row[0:32])
    statistics_hiphop.append(statistics_hip[0])
    statistics_hiphop.extend(statistics_hip[1:101])
    statistics_hiphop = preprocessing.scale(statistics_hiphop)

#with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_jazz.csv", "r") as statisticsja:
with open("C:\Users\David\Desktop\TFG\\10_Train_Test_Externa\\5_Descriptors_Statistics_Data_Test\Statistics_jazz.csv", "r") as statisticsja:
    statisticsja.readline()
    statisticsja = csv.reader(statisticsja)
    for row in statisticsja:
        row = map(float,row[0:32])
        statistics_ja.append(row[0:32])
    statistics_jazz.append(statistics_ja[0])
    statistics_jazz.extend(statistics_ja[1:101])
    statistics_jazz = preprocessing.scale(statistics_jazz)

##with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_metal.csv", "r") as statisticsme:
##    statisticsme.readline()
##    statisticsme = csv.reader(statisticsme)
##    for row in statisticsme:
##        row = map(float,row[0:32])
##        statistics_me.append(row[0:32])
##    statistics_metal.append(statistics_me[0])
##    statistics_metal.extend(statistics_me[1:101])
##    statistics_metal = preprocessing.scale(statistics_metal)

#with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_pop.csv", "r") as statisticspo:
with open("C:\Users\David\Desktop\TFG\\10_Train_Test_Externa\\5_Descriptors_Statistics_Data_Test\Statistics_pop.csv", "r") as statisticspo:
    statisticspo.readline()
    statisticspo = csv.reader(statisticspo)
    for row in statisticspo:
        row = map(float,row[0:32])
        statistics_po.append(row[0:32])
    statistics_pop.append(statistics_po[0])
    statistics_pop.extend(statistics_po[1:101])
    statistics_pop = preprocessing.scale(statistics_pop)

##with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_reggae.csv", "r") as statisticsreg:
##    statisticsreg.readline()
##    statisticsreg = csv.reader(statisticsreg)
##    for row in statisticsreg:
##        row = map(float,row[0:32])
##        statistics_reg.append(row[0:32])
##    statistics_reggae.append(statistics_reg[0])
##    statistics_reggae.extend(statistics_reg[1:101])
##    statistics_reggae = preprocessing.scale(statistics_reggae)

#with open("C:\Users\David\Desktop\TFG\\5_Descriptors_Statistics_Data\Statistics_rock.csv", "r") as statisticsro:
with open("C:\Users\David\Desktop\TFG\\10_Train_Test_Externa\\5_Descriptors_Statistics_Data_Test\Statistics_rock.csv", "r") as statisticsro:
    statisticsro.readline()
    statisticsro = csv.reader(statisticsro)
    for row in statisticsro:
        row = map(float,row[0:32])
        statistics_ro.append(row[0:32])
    statistics_rock.append(statistics_ro[0])
    statistics_rock.extend(statistics_ro[1:101])
    statistics_rock = preprocessing.scale(statistics_rock)

#----------------------------------------------------Products------------------------------------------------------
#All Genres with Eigen Vectors Blues
blues_pcablues =[[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_blues]
#classical_pcablues = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_classical]
country_pcablues = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_country]
#disco_pcablues = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_disco]
hiphop_pcablues = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_hiphop]
jazz_pcablues = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_jazz]
#metal_pcablues = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_metal]
pop_pcablues = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_pop]
#reggae_pcablues =[[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_reggae]
rock_pcablues = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_blues)] for X_row in statistics_rock]


#####All Genres with Eigen Vectors Classical
##blues_pcaclassical =[[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_blues]
##classical_pcaclassical = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_classical]
##country_pcaclassical = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_country]
##disco_pcaclassical = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_disco]
##hiphop_pcaclassical = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_hiphop]
##jazz_pcaclassical = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_jazz]
##metal_pcaclassical = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_metal]
##pop_pcaclassical = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_pop]
##reggae_pcaclassical = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_reggae]
##rock_pcaclassical = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_classical)] for X_row in statistics_rock]
##
#All Genres with Eigen Vectors Country
blues_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_blues]
#classical_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_classical]
country_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_country]
#disco_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_disco]
hiphop_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_hiphop]
jazz_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_jazz]
#metal_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_metal]
pop_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_pop]
#reggae_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_reggae]
rock_pcacountry = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_country)] for X_row in statistics_rock]

#All Genres with Eigen Vectors Disco
##blues_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_blues]
##classical_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_classical]
##country_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_country]
##disco_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_disco]
##hiphop_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_hiphop]
##jazz_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_jazz]
##metal_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_metal]
##pop_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_pop]
##reggae_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_reggae]
##rock_pcadisco = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_disco)] for X_row in statistics_rock]

#All Genres with Eigen Vectors Hiphop
blues_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_blues]
#classical_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_classical]
country_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_country]
#disco_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_disco]
hiphop_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_hiphop]
jazz_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_jazz]
#metal_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_metal]
pop_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_pop]
#reggae_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_reggae]
rock_pcahiphop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_hiphop)] for X_row in statistics_rock]

#All Genres with Eigen Vectors Jazz
blues_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_blues]
#classical_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_classical]
country_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_country]
#disco_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_disco]
hiphop_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_hiphop]
jazz_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_jazz]
#metal_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_metal]
pop_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_pop]
#reggae_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_reggae]
rock_pcajazz = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_jazz)] for X_row in statistics_rock]

#All Genres with Eigen Vectors Metal
##blues_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_blues]
##classical_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_classical]
##country_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_country]
##disco_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_disco]
##hiphop_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_hiphop]
##jazz_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_jazz]
##metal_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_metal]
##pop_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_pop]
##reggae_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_reggae]
##rock_pcametal = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_metal)] for X_row in statistics_rock]

#All Genres with Eigen Vectors Pop
blues_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_blues]
#classical_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_classical]
country_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_country]
#disco_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_disco]
hiphop_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_hiphop]
jazz_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_jazz]
#metal_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_metal]
pop_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_pop]
#reggae_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_reggae]
rock_pcapop = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_pop)] for X_row in statistics_rock]

#All Genres with Eigen Vectors Reggae
##blues_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_blues]
##classical_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_classical]
##country_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_country]
##disco_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_disco]
##hiphop_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_hiphop]
##jazz_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_jazz]
##metal_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_metal]
##pop_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_pop]
##reggae_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_reggae]
##rock_pcareggae = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_reggae)] for X_row in statistics_rock]

#All Genres with Eigen Vectors Rock
blues_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_blues]
#classical_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_classical]
country_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_country]
#disco_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_disco]
hiphop_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_hiphop]
jazz_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_jazz]
#metal_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_metal]
pop_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_pop]
#reggae_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_reggae]
rock_pcarock = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*eigenvectors_rock)] for X_row in statistics_rock]



for statisticFile in statisticsList:
    statisticFilePath = (os.path.join(statisticsDataFolder,statisticFile))
    print (statisticFilePath)

    if statisticFile.endswith("blues.csv"):
        for row in blues_pcablues:
            row.append("blu")
##        for row in classical_pcablues:
##            row.append("no blu")
        for row in country_pcablues:
            row.append("no blu")
##        for row in disco_pcablues:
##            row.append("no blu")
        for row in hiphop_pcablues:
            row.append("no blu")
        for row in jazz_pcablues:
            row.append("no blu")
##        for row in metal_pcablues:
##            row.append("no blu")
        for row in pop_pcablues:
            row.append("no blu")
##        for row in reggae_pcablues:
##            row.append("no blu")
        for row in rock_pcablues:
            row.append("no blu")

        lenblues = len(blues_pcablues[0])
        header_data_blues_no_blues = []
        for i in range(1,lenblues):
            header_data_blues_no_blues.append("PC" + str(i))
            for j in header_data_blues_no_blues:
                exec("{var}=[]".format(var=j))
        header_data_blues_no_blues.append("Genre")

        data_blues_no_blues.append(header_data_blues_no_blues)    
        data_blues_no_blues.extend(blues_pcablues)
        #data_blues_no_blues.extend(classical_pcablues)
        data_blues_no_blues.extend(country_pcablues)
        #data_blues_no_blues.extend(disco_pcablues)
        data_blues_no_blues.extend(hiphop_pcablues)
        data_blues_no_blues.extend(jazz_pcablues)
        #data_blues_no_blues.extend(metal_pcablues)
        data_blues_no_blues.extend(pop_pcablues)
        #data_blues_no_blues.extend(reggae_pcablues)
        data_blues_no_blues.extend(rock_pcablues)

        write_file = "single_classifier_blues_no_blues.csv"
        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
            csv_single_classifier_blues_no_blues = csv.writer(csvfile)
            for row in data_blues_no_blues:
                csv_single_classifier_blues_no_blues.writerows([row])

##    if statisticFile.endswith("classical.csv"):
##        for row in classical_pcaclassical:
##            row.append("cla")
##        for row in blues_pcaclassical:
##            row.append("no cla")
##        for row in country_pcaclassical:
##            row.append("no cla")
##        for row in disco_pcaclassical:
##            row.append("no cla")
##        for row in hiphop_pcaclassical:
##            row.append("no cla")
##        for row in jazz_pcaclassical:
##            row.append("no cla")
##        for row in metal_pcaclassical:
##            row.append("no cla")
##        for row in pop_pcaclassical:
##            row.append("no cla")
##        for row in reggae_pcaclassical:
##            row.append("no cla")
##        for row in rock_pcaclassical:
##            row.append("no cla")
##
##        lenclass = len(classical_pcaclassical[0])
##        header_data_classical_no_classical = []
##        for i in range(1,lenclass):
##            header_data_classical_no_classical.append("PC" + str(i))
##            for j in  header_data_classical_no_classical:
##                exec("{var}=[]".format(var=j))
##        header_data_classical_no_classical.append("Genre")
##
##        data_classical_no_classical.append(header_data_classical_no_classical) 
##        data_classical_no_classical.extend(classical_pcaclassical)
##        data_classical_no_classical.extend(blues_pcaclassical)
##        data_classical_no_classical.extend(country_pcaclassical)
##        data_classical_no_classical.extend(disco_pcaclassical)
##        data_classical_no_classical.extend(hiphop_pcaclassical)
##        data_classical_no_classical.extend(jazz_pcaclassical)
##        data_classical_no_classical.extend(metal_pcaclassical)
##        data_classical_no_classical.extend(pop_pcaclassical)
##        data_classical_no_classical.extend(reggae_pcaclassical)
##        data_classical_no_classical.extend(rock_pcaclassical)
##
##        write_file = "single_classifier_classical_no_classical.csv"
##        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
##            csv_single_classifier_classical_no_classical = csv.writer(csvfile)
##            for row in data_classical_no_classical:
##                csv_single_classifier_classical_no_classical.writerows([row])

    if statisticFile.endswith("country.csv"):
        for row in country_pcacountry:
            row.append("count")
        for row in blues_pcacountry:
            row.append("no count")
##        for row in classical_pcacountry:
##            row.append("no count")
##        for row in disco_pcacountry:
##            row.append("no count")
        for row in hiphop_pcacountry:
            row.append("no count")
        for row in jazz_pcacountry:
            row.append("no count")
##        for row in metal_pcacountry:
##            row.append("no count")
        for row in pop_pcacountry:
            row.append("no count")
##        for row in reggae_pcacountry:
##            row.append("no count")
        for row in rock_pcacountry:
            row.append("no count")

        lencount = len(country_pcacountry[0])
        header_data_country_no_country = []
        for i in range(1,lencount):
            header_data_country_no_country.append("PC" + str(i))
            for j in  header_data_country_no_country:
                exec("{var}=[]".format(var=j))
        header_data_country_no_country.append("Genre")

        data_country_no_country.append(header_data_country_no_country)
        data_country_no_country.extend(country_pcacountry)
        data_country_no_country.extend(blues_pcacountry)
        #data_country_no_country.extend(classical_pcacountry)
        #data_country_no_country.extend(disco_pcacountry)
        data_country_no_country.extend(hiphop_pcacountry)
        data_country_no_country.extend(jazz_pcacountry)
        #data_country_no_country.extend(metal_pcacountry)
        data_country_no_country.extend(pop_pcacountry)
        #data_country_no_country.extend(reggae_pcacountry)
        data_country_no_country.extend(rock_pcacountry)
              

        write_file = "single_classifier_country_no_country.csv"
        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
            csv_single_classifier_country_no_country = csv.writer(csvfile)
            for row in data_country_no_country:
                csv_single_classifier_country_no_country.writerows([row])

##    if statisticFile.endswith("disco.csv"):
##        for row in disco_pcadisco:
##            row.append("disco")
##        for row in blues_pcadisco:
##            row.append("no disco")
##        for row in classical_pcadisco:
##            row.append("no disco")
##        for row in country_pcadisco:
##            row.append("no disco")
##        for row in hiphop_pcadisco:
##            row.append("no disco")
##        for row in jazz_pcadisco:
##            row.append("no disco")
##        for row in metal_pcadisco:
##            row.append("no disco")
##        for row in pop_pcadisco:
##            row.append("no disco")
##        for row in reggae_pcadisco:
##            row.append("no disco")
##        for row in rock_pcadisco:
##            row.append("no disco")
##
##        lendisc = len(disco_pcadisco[0])
##        header_data_disco_no_disco = []
##        for i in range(1,lendisc):
##            header_data_disco_no_disco.append("PC" + str(i))
##            for j in  header_data_disco_no_disco:
##                exec("{var}=[]".format(var=j))
##        header_data_disco_no_disco.append("Genre")
##
##        data_disco_no_disco.append(header_data_disco_no_disco)
##        data_disco_no_disco.extend(disco_pcadisco)
##        data_disco_no_disco.extend(blues_pcadisco)
##        data_disco_no_disco.extend(classical_pcadisco)
##        data_disco_no_disco.extend(country_pcadisco)
##        data_disco_no_disco.extend(hiphop_pcadisco)
##        data_disco_no_disco.extend(jazz_pcadisco)
##        data_disco_no_disco.extend(metal_pcadisco)
##        data_disco_no_disco.extend(pop_pcadisco)
##        data_disco_no_disco.extend(reggae_pcadisco)
##        data_disco_no_disco.extend(rock_pcadisco)
##                      
##
##        write_file = "single_classifier_disco_no_disco.csv"
##        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
##            csv_single_classifier_disco_no_disco = csv.writer(csvfile)
##            for row in data_disco_no_disco:
##                csv_single_classifier_disco_no_disco.writerows([row])

    if statisticFile.endswith("hiphop.csv"):
        for row in hiphop_pcahiphop:
            row.append("hiphop")
        for row in blues_pcahiphop:
            row.append("no hiphop")
##        for row in classical_pcahiphop:
##            row.append("no hiphop")
        for row in country_pcahiphop:
            row.append("no hiphop")
##        for row in disco_pcahiphop:
##            row.append("no hiphop")
        for row in jazz_pcahiphop:
            row.append("no hiphop")
##        for row in metal_pcahiphop:
##            row.append("no hiphop")
        for row in pop_pcahiphop:
            row.append("no hiphop")
##        for row in reggae_pcahiphop:
##            row.append("no hiphop")
        for row in rock_pcahiphop:
            row.append("no hiphop")

        lenhip = len(hiphop_pcahiphop[0])
        header_data_hiphop_no_hiphop = []
        for i in range(1,lenhip):
            header_data_hiphop_no_hiphop.append("PC" + str(i))
            for j in  header_data_hiphop_no_hiphop:
                exec("{var}=[]".format(var=j))
        header_data_hiphop_no_hiphop.append("Genre")
        
        data_hiphop_no_hiphop.append(header_data_hiphop_no_hiphop)
        data_hiphop_no_hiphop.extend(hiphop_pcahiphop)
        data_hiphop_no_hiphop.extend(blues_pcahiphop)
        #data_hiphop_no_hiphop.extend(classical_pcahiphop)
        data_hiphop_no_hiphop.extend(country_pcahiphop)
        #data_hiphop_no_hiphop.extend(disco_pcahiphop)
        data_hiphop_no_hiphop.extend(jazz_pcahiphop)
        #data_hiphop_no_hiphop.extend(metal_pcahiphop)
        data_hiphop_no_hiphop.extend(pop_pcahiphop)
        #data_hiphop_no_hiphop.extend(reggae_pcahiphop)
        data_hiphop_no_hiphop.extend(rock_pcahiphop)
                      

        write_file = "single_classifier_hiphop_no_hiphop.csv"
        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
            csv_single_classifier_hiphop_no_hiphop = csv.writer(csvfile)
            for row in data_hiphop_no_hiphop:
                csv_single_classifier_hiphop_no_hiphop.writerows([row])

    if statisticFile.endswith("jazz.csv"):
        for row in jazz_pcajazz:
            row.append("jazz")
        for row in blues_pcajazz:
            row.append("no jazz")
##        for row in classical_pcajazz:
##            row.append("no jazz")
        for row in country_pcajazz:
            row.append("no jazz")
##        for row in disco_pcajazz:
##            row.append("no jazz")
        for row in hiphop_pcajazz:
            row.append("no jazz")
##        for row in metal_pcajazz:
##            row.append("no jazz")
        for row in pop_pcajazz:
            row.append("no jazz")
##        for row in reggae_pcajazz:
##            row.append("no jazz")
        for row in rock_pcajazz:
            row.append("no jazz")

        lenjazz = len(jazz_pcajazz[0])
        header_data_jazz_no_jazz = []
        for i in range(1,lenjazz):
            header_data_jazz_no_jazz.append("PC" + str(i))
            for j in  header_data_jazz_no_jazz:
                exec("{var}=[]".format(var=j))
        header_data_jazz_no_jazz.append("Genre")

        data_jazz_no_jazz.append(header_data_jazz_no_jazz)
        data_jazz_no_jazz.extend(jazz_pcajazz)
        data_jazz_no_jazz.extend(blues_pcajazz)
        #data_jazz_no_jazz.extend(classical_pcajazz)
        data_jazz_no_jazz.extend(country_pcajazz)
        #data_jazz_no_jazz.extend(disco_pcajazz)
        data_jazz_no_jazz.extend(hiphop_pcajazz)
        #data_jazz_no_jazz.extend(metal_pcajazz)
        data_jazz_no_jazz.extend(pop_pcajazz)
        #data_jazz_no_jazz.extend(reggae_pcajazz)
        data_jazz_no_jazz.extend(rock_pcajazz)
                      

        write_file = "single_classifier_jazz_no_jazz.csv"
        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
            csv_single_classifier_jazz_no_jazz = csv.writer(csvfile)
            for row in data_jazz_no_jazz:
                csv_single_classifier_jazz_no_jazz.writerows([row])
##
##    if statisticFile.endswith("metal.csv"):
##        for row in metal_pcametal:
##            row.append("metal")
##        for row in blues_pcametal:
##            row.append("no metal")
##        for row in classical_pcametal:
##            row.append("no metal")
##        for row in country_pcametal:
##            row.append("no metal")
##        for row in disco_pcametal:
##            row.append("no metal")
##        for row in hiphop_pcametal:
##            row.append("no metal")
##        for row in jazz_pcametal:
##            row.append("no metal")
##        for row in pop_pcametal:
##            row.append("no metal")
##        for row in reggae_pcametal:
##            row.append("no metal")
##        for row in rock_pcametal:
##            row.append("no metal")
##
##        lenmetal = len(metal_pcametal[0])
##        header_data_metal_no_metal = []
##        for i in range(1,lenmetal):
##            header_data_metal_no_metal.append("PC" + str(i))
##            for j in  header_data_metal_no_metal:
##                exec("{var}=[]".format(var=j))
##        header_data_metal_no_metal.append("Genre")
##
##        data_metal_no_metal.append(header_data_metal_no_metal)
##        data_metal_no_metal.extend(metal_pcametal)
##        data_metal_no_metal.extend(blues_pcametal)
##        data_metal_no_metal.extend(classical_pcametal)
##        data_metal_no_metal.extend(country_pcametal)
##        data_metal_no_metal.extend(disco_pcametal)
##        data_metal_no_metal.extend(hiphop_pcametal)
##        data_metal_no_metal.extend(jazz_pcametal)
##        data_metal_no_metal.extend(pop_pcametal)
##        data_metal_no_metal.extend(reggae_pcametal)
##        data_metal_no_metal.extend(rock_pcametal)
##                      
##
##        write_file = "single_classifier_metal_no_metal.csv"
##        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
##            csv_single_classifier_metal_no_metal = csv.writer(csvfile)
##            for row in data_metal_no_metal:
##                csv_single_classifier_metal_no_metal.writerows([row])

    if statisticFile.endswith("pop.csv"):
        for row in pop_pcapop:
            row.append("pop")
        for row in blues_pcapop:
            row.append("no pop")
##        for row in classical_pcapop:
##            row.append("no pop")
        for row in country_pcapop:
            row.append("no pop")
##        for row in disco_pcapop:
##            row.append("no pop")
        for row in hiphop_pcapop:
            row.append("no pop")
        for row in jazz_pcapop:
            row.append("no pop")
##        for row in metal_pcapop:
##            row.append("no pop")
##        for row in reggae_pcapop:
##            row.append("no pop")
        for row in rock_pcapop:
            row.append("no pop")

        lenpop = len(pop_pcapop[0])
        header_data_pop_no_pop = []
        for i in range(1,lenpop):
            header_data_pop_no_pop.append("PC" + str(i))
            for j in  header_data_pop_no_pop:
                exec("{var}=[]".format(var=j))
        header_data_pop_no_pop.append("Genre")

        data_pop_no_pop.append(header_data_pop_no_pop)
        data_pop_no_pop.extend(pop_pcapop)
        data_pop_no_pop.extend(blues_pcapop)
        #data_pop_no_pop.extend(classical_pcapop)
        data_pop_no_pop.extend(country_pcapop)
        #data_pop_no_pop.extend(disco_pcapop)
        data_pop_no_pop.extend(hiphop_pcapop)
        data_pop_no_pop.extend(jazz_pcapop)
        #data_pop_no_pop.extend(metal_pcapop)
        #data_pop_no_pop.extend(reggae_pcapop)
        data_pop_no_pop.extend(rock_pcapop)
                      

        write_file = "single_classifier_pop_no_pop.csv"
        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
            csv_single_classifier_pop_no_pop = csv.writer(csvfile)
            for row in data_pop_no_pop:
                csv_single_classifier_pop_no_pop.writerows([row])

##    if statisticFile.endswith("reggae.csv"):
##        for row in reggae_pcareggae:
##            row.append("reggae")
##        for row in blues_pcareggae:
##            row.append("no reggae")
##        for row in classical_pcareggae:
##            row.append("no reggae")
##        for row in country_pcareggae:
##            row.append("no reggae")
##        for row in disco_pcareggae:
##            row.append("no reggae")
##        for row in hiphop_pcareggae:
##            row.append("no reggae")
##        for row in jazz_pcareggae:
##            row.append("no reggae")
##        for row in metal_pcareggae:
##            row.append("no reggae")
##        for row in pop_pcareggae:
##            row.append("no reggae")
##        for row in rock_pcareggae:
##            row.append("no reggae")
##
##        lenreg = len(reggae_pcareggae[0])
##        header_data_reggae_no_reggae = []
##        for i in range(1,lenreg):
##            header_data_reggae_no_reggae.append("PC" + str(i))
##            for j in   header_data_reggae_no_reggae:
##                exec("{var}=[]".format(var=j))
##        header_data_reggae_no_reggae.append("Genre")
##
##        data_reggae_no_reggae.append(header_data_reggae_no_reggae)
##        data_reggae_no_reggae.extend(reggae_pcareggae)
##        data_reggae_no_reggae.extend(blues_pcareggae)
##        data_reggae_no_reggae.extend(classical_pcareggae)
##        data_reggae_no_reggae.extend(country_pcareggae)
##        data_reggae_no_reggae.extend(disco_pcareggae)
##        data_reggae_no_reggae.extend(hiphop_pcareggae)
##        data_reggae_no_reggae.extend(jazz_pcareggae)
##        data_reggae_no_reggae.extend(metal_pcareggae)
##        data_reggae_no_reggae.extend(pop_pcareggae)
##        data_reggae_no_reggae.extend(rock_pcareggae)
##                      
##
##        write_file = "single_classifier_reggae_no_reggae.csv"
##        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
##            csv_single_classifier_reggae_no_reggae = csv.writer(csvfile)
##            for row in data_reggae_no_reggae:
##                csv_single_classifier_reggae_no_reggae.writerows([row])

    if statisticFile.endswith("rock.csv"):
        for row in rock_pcarock:
            row.append("rock")
        for row in blues_pcarock:
            row.append("no rock")
##        for row in classical_pcarock:
##            row.append("no rock")
        for row in country_pcarock:
            row.append("no rock")
##        for row in disco_pcarock:
##            row.append("no rock")
        for row in hiphop_pcarock:
            row.append("no rock")
        for row in jazz_pcarock:
            row.append("no rock")
##        for row in metal_pcarock:
##            row.append("no rock")
        for row in pop_pcarock:
            row.append("no rock")
##        for row in reggae_pcarock:
##            row.append("no rock")

        lenroc = len(rock_pcarock[0])
        header_data_rock_no_rock = []
        for i in range(1,lenroc):
            header_data_rock_no_rock.append("PC" + str(i))
            for j in   header_data_rock_no_rock:
                exec("{var}=[]".format(var=j))
        header_data_rock_no_rock.append("Genre")

        data_rock_no_rock.append( header_data_rock_no_rock)
        data_rock_no_rock.extend(rock_pcarock)
        data_rock_no_rock.extend(blues_pcarock)
        #data_rock_no_rock.extend(classical_pcarock)
        data_rock_no_rock.extend(country_pcarock)
        #data_rock_no_rock.extend(disco_pcarock)
        data_rock_no_rock.extend(hiphop_pcarock)
        data_rock_no_rock.extend(jazz_pcarock)
        #data_rock_no_rock.extend(metal_pcarock)
        data_rock_no_rock.extend(pop_pcarock)
        #data_rock_no_rock.extend(reggae_pcarock)
                      

        write_file = "single_classifier_rock_no_rock.csv"
        with open(singleClassifierDataSaving +"\\"+write_file, "w") as csvfile:
            csv_single_classifier_rock_no_rock = csv.writer(csvfile)
            for row in data_rock_no_rock:
                csv_single_classifier_rock_no_rock.writerows([row])                
                   
print "Computing Done"
jvm.stop()



