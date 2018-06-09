#IMPORTS
import sys
import os
import csv
import itertools
import numpy
from javabridge import JWrapper, JClassWrapper
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
import weka.core.jvm as jvm

jvm.start()

#INFO
print ("---------------------------------Info-------------------------------------")
print ("This is a script that load data to Weka, perform Atribute Selection based on PCA and save EigenVectors Matrix in a .csv file")
print ("                                                                                                                            ")
print ("David Campayo, Final Degree Project, Audiovisual Systems Engineering, 2018                                                  ")
print ("----------------------------------------------------------------------------------------------------------------------------")
print ("                                                            ")

#READING COMMAND LINE

if len(sys.argv) < 3:
    print ("----------------------------- How to use it-------------------------------")
    print ("                                                                 ")
    print ("Declaration:            -> doPCA.py path\\to\\data\\folder path\\to\\data\\saving Atribute Evaluator")
    print ("Available Atribute Evaluator:                       ")
    print ("                        -> CsfSubsetEval")
    print ("                        -> PrincipalComponents")
    print ("                        -> WrapperSubsetEval")

    sys.exit(-1)


#DECLARATION DATA
statisticsDataFolder = sys.argv[1]
pcaDataSaving = sys.argv[2]
atributeEvaluator = sys.argv[3]
row_count = 0
eigenvectorsDataFolder = "C:\\Users\\David\\Desktop\\TFG\\7_EigenVectors"

header_eigenvector = []
previous_eigen_vectors = []
eigen_vectors_matrix = []

#1. Searching Files

statisticsList = os.listdir(statisticsDataFolder)
print ("This is your statistics database:  ")
print ("                                   ")
print statisticsList

for statisticFile in statisticsList:
    statisticFilePath = (os.path.join(statisticsDataFolder,statisticFile))
    print (statisticFilePath)

    if statisticFile.endswith("blues.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_blues.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_blues.csv","r") as blueseigenvectors:
            blueseigenvectors1 = csv.reader(blueseigenvectors)
            for row in blueseigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_blues.csv","r") as blueseigenvectors:
                            blueseigenvectors1 = csv.reader(blueseigenvectors)
                            row = list(blueseigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_blues.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

    if statisticFile.endswith("classical.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_classical.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_classical.csv","r") as classicaleigenvectors:
            classicaleigenvectors1 = csv.reader(classicaleigenvectors)
            for row in classicaleigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_classical.csv","r") as classicaleigenvectors:
                            classicaleigenvectors1 = csv.reader(classicaleigenvectors)
                            row = list(classicaleigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_classical.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

    if statisticFile.endswith("country.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_country.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_country.csv","r") as countryeigenvectors:
            countryeigenvectors1 = csv.reader(countryeigenvectors)
            for row in countryeigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_country.csv","r") as countryeigenvectors:
                            countryeigenvectors1 = csv.reader(countryeigenvectors)
                            row = list(countryeigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_country.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

    if statisticFile.endswith("disco.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_disco.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_disco.csv","r") as discoeigenvectors:
            discoeigenvectors1 = csv.reader(discoeigenvectors)
            for row in discoeigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_disco.csv","r") as discoeigenvectors:
                            discoeigenvectors1 = csv.reader(discoeigenvectors)
                            row = list(discoeigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_disco.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

    if statisticFile.endswith("hiphop.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_hiphop.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_hiphop.csv","r") as hiphopeigenvectors:
            hiphopeigenvectors1 = csv.reader(hiphopeigenvectors)
            for row in hiphopeigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_hiphop.csv","r") as hiphopeigenvectors:
                            hiphopeigenvectors1 = csv.reader(hiphopeigenvectors)
                            row = list(hiphopeigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_hiphop.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

    if statisticFile.endswith("jazz.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_jazz.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_jazz.csv","r") as jazzeigenvectors:
            jazzeigenvectors1 = csv.reader(jazzeigenvectors)
            for row in jazzeigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_jazz.csv","r") as jazzeigenvectors:
                            jazzeigenvectors1 = csv.reader(jazzeigenvectors)
                            row = list(jazzeigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_jazz.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

    if statisticFile.endswith("metal.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_metal.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_metal.csv","r") as metaleigenvectors:
            metaleigenvectors1 = csv.reader(metaleigenvectors)
            for row in metaleigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_metal.csv","r") as metaleigenvectors:
                            metaleigenvectors1 = csv.reader(metaleigenvectors)
                            row = list(metaleigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_metal.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

    if statisticFile.endswith("pop.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_pop.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_pop.csv","r") as popeigenvectors:
            popeigenvectors1 = csv.reader(popeigenvectors)
            for row in popeigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_pop.csv","r") as popeigenvectors:
                            popeigenvectors1 = csv.reader(popeigenvectors)
                            row = list(popeigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_pop.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

    if statisticFile.endswith("reggae.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_reggae.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_reggae.csv","r") as reggaeeigenvectors:
            reggaeeigenvectors1 = csv.reader(reggaeeigenvectors)
            for row in reggaeeigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_reggae.csv","r") as reggaeeigenvectors:
                            reggaeeigenvectors1 = csv.reader(reggaeeigenvectors)
                            row = list(reggaeeigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_reggae.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

    if statisticFile.endswith("rock.csv"):
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(statisticFilePath)
        data.class_is_last()
        print data

        if atributeEvaluator == "PrincipalComponents" :
            evaluator = ASEvaluation(classname="weka.attributeSelection.PrincipalComponents")
            search = ASSearch(classname="weka.attributeSelection.Ranker")
            attsel = AttributeSelection()
            attsel.search(search)
            attsel.evaluator(evaluator)
            print (attsel.select_attributes(data))
            print("# attributes: " + str(attsel.number_attributes_selected))
            num_attributes = attsel.number_attributes_selected
            print("attributes: " + str(attsel.selected_attributes))
            y = attsel.selected_attributes
            #print("result string:\n" + attsel.results_string)
            results_pca = attsel.results_string

        write_file = "PCA_rock.csv"
        with open(pcaDataSaving +"\\"+write_file, "w") as output:
            for line in results_pca:
                output.write(line)

        #Creating csv with Eigenvectors:
        #Headers: V1 to length of # of eigenvectors (z) and empty EigenVectors arrays
##        for i in range(1,num_attributes+1):
##            header_eigenvector.append("V" + str(i))
##            for j in header_eigenvector:
##                exec("{var}=[]".format(var=j))
##        print header_eigenvector

        #Adding Eigenvectors Value to create a matrix
        name = "Eigenvectors"
        with open(pcaDataSaving+"\\"+"PCA_rock.csv","r") as rockeigenvectors:
            rockeigenvectors1 = csv.reader(rockeigenvectors)
            for row in rockeigenvectors1:
                row_count = row_count + 1
                for field in row:
                    if field == name:
                        #row_count = row_count + 1
                        print row_count
                        with open(pcaDataSaving+"\\"+"PCA_rock.csv","r") as rockeigenvectors:
                            rockeigenvectors1 = csv.reader(rockeigenvectors)
                            row = list(rockeigenvectors1)
                            for i in range(row_count,row_count+33):   
                                previous_eigen_vectors = "".join(row[i])
                                previous_eigen_vectors = previous_eigen_vectors.replace("\t",",")
                                previous_eigen_vectors = previous_eigen_vectors.split(',')
                                eigen_vectors_matrix.append(previous_eigen_vectors)
                            print eigen_vectors_matrix
        with open(eigenvectorsDataFolder +"\\"+ "EigenVectors_rock.csv", "w") as eigenvectors_csvfile:
            writer = csv.writer(eigenvectors_csvfile)
            #writer.writerow(header_eigenvector)
            writer.writerows(eigen_vectors_matrix)
        row_count = 0
        previous_eigen_vectors = []
        eigen_vectors_matrix = []

print "Computing Done"
jvm.stop()


