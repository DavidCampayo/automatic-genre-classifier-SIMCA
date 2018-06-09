#INFO: This is a script that calculates audio statistics from calculated descriptors using doDescriptors.py
#and obtain .csv file ready to be used in Weka


#vamp:beatroot-vamp:beatroot:beats
#--->vamp:mir-edu:attackstartendtimes:attackstartendtimes
#vamp:mir-edu:logattacktime:logattacktime
#vamp:mir-edu:rms:rms
#vamp:mir-edu:spectralcentroid:spectralcentroid
#vamp:mir-edu:spectralcrest:spectralcrest
#vamp:mir-edu:spectralflatness:spectralflatness
#vamp:mir-edu:spectralflux:spectralflux
#vamp:mir-edu:spectralkurtosis:spectralkurtosis
#vamp:mir-edu:spectralrolloff:spectralrolloff
#vamp:mir-edu:spectralskewness:spectralskewness
#vamp:mir-edu:spectralspread:spectralspread
#vamp:mir-edu:temporalcentroid:temporalcentroid
#vamp:mir-edu:zerocrossingrate:zerocrossingrate


#IMPORTS
import sys
import os
import csv
import numpy

#INFO
print ("---------------------------------------------Info---------------------------------------------")
print ("This is a script that calculates Audio Statistics and makes .csv file ready to be used in Weka")
print ("                                                                                              ")
print ("David Campayo, Final Degree Project, Audiovisual Systems Engineering, 2018")
print ("----------------------------------------------------------------------------------------------")
print ("                                                                                              ")

#READING COMMAND LINE

if len(sys.argv) < 4:
    print ("----------------------------- How to use it-------------------------------")
    print ("                                                                 ")
    print ("Declaration:            -> doStatistics.py path\\to\\descriptor\\data\\folder path\\to\\statistics\\data\\saving descriptor")
    print ("Available Descriptors:                       ")
    print ("                        -> beats")
    print ("                        -> attacksartendtimes")
    print ("                        -> logattacktime")
    print ("                        -> rms")
    print ("                        -> spectralcentroid")  
    print ("                        -> spectralcrest")  
    print ("                        -> spectralflatness")
    print ("                        -> spectralflux")
    print ("                        -> spectralkurtosis")
    print ("                        -> spectralrolloff")
    print ("                        -> spectralskewness")
    print ("                        -> spectralspread")
    print ("                        -> temporalcentroid")
    print ("                        -> zerocrossingrate")
    print ("                        -> mfcc")
    sys.exit(-1)


#DECLARATION DATA
descriptorDataFolder = sys.argv[1]
statisticsDataSaving = sys.argv[2]
descriptorUsed = sys.argv[3]

headers = []
dataStatistics = []
genre = []
totalData = []

#DECLARATION OF DESCRIPTORS DATA AND STATISTICS
dataBeats = []
meanDataBeats = []
varDataBeats = []

dataAttackStartEndTimes =[]
meanDataAttackStartEndTimes = []
varDataAttackStartEndTimes = []

dataLogAttackTime = []
meanDataLogAttackTime = []
varDataLogAttackTime = []

dataRms = []
meanDataRms = []
varDataRms = []

dataSpectralCentroid = []
meanDataSpectralCentroid = []
varDataSpectralCentroid = []

dataSpectralCrest = []
meanDataSpectralCrest = []
varDataSpectralCrest = []

dataSpectralFlatness = []
meanDataSpectralFlatness = []
varDataSpectralFlatness = []

dataSpectralFlux = []
meanDataSpectralFlux = []
varDataSpectralFlux = []

dataSpectralKurtosis = []
meanDataSpectralKurtosis = []
varDataSpectralKurtosis = []

dataSpectralRollOff = []
meanDataSpectralRollOff = []
varDataSpectralRollOff = []

dataSpectralSkewness = []
meanDataSpectralSkewness = []
varDataSpectralSkewness = []

dataSpectralSpread = []
meanDataSpectralSpread= []
varDataSpectralSpread = []

dataTemporalCentroid = []
meanDataTemporalCentroid = []
varDataTemporalCentroid = []

dataZeroCrossingRate = []
meanDataZeroCrossingRate = []
varDataZeroCrossingRate = []

dataMfcc0 = []
meanDataMfcc0 = []
dataMfcc1 = []
meanDataMfcc1 = []
dataMfcc2 = []
meanDataMfcc2 = []
dataMfcc3 = []
meanDataMfcc0 = []
dataMfcc0 = []
meanDataMfcc3 = []
dataMfcc4 = []
meanDataMfcc4 = []
dataMfcc5 = []
meanDataMfcc5 = []
dataMfcc6 = []
meanDataMfcc6 = []
dataMfcc7 = []
meanDataMfcc7 = []
dataMfcc8 = []
meanDataMfcc8 = []
dataMfcc9 = []
meanDataMfcc9 = []
dataMfcc10 = []
meanDataMfcc10 = []
dataMfcc11 = []
meanDataMfcc11 = []
dataMfcc12 = []
meanDataMfcc12 = []
dataMfcc13 = []
meanDataMfcc13 = []
dataMfcc14 = []
meanDataMfcc14 = []
dataMfcc15 = []
meanDataMfcc15 = []
dataMfcc16 = []
meanDataMfcc16 = []
dataMfcc17 = []
meanDataMfcc17 = []
dataMfcc18 = []
meanDataMfcc18 = []
dataMfcc19 = []
meanDataMfcc19 = []
dataMfcc20 = []
meanDataMfcc20 = []



#DECLATION COUNTERS

counter_beats = 0
counter_attackstartendtimes = 0
counter_logattacktime = 0
counter_rms = 0
counter_spectralcentroid = 0
counter_spectralcrest = 0
counter_spectralflatness = 0
counter_meanspectralflatness = 0
counter_spectralflux = 0
counter_spectralkurtosis = 0
counter_spectralrolloff = 0
counter_spectralskewness = 0
counter_spectralspread = 0
counter_temporalcentroid = 0
counter_zerocrossingrate = 0
counter_mfcc = 0

#DECLARATION PATH SONIC-ANNOTATOR
sonic_annotator = "C:\Users\David\Desktop\Sonic-Annotator\sonic-annotator-v1.5-win32"

#COMPUTING
#1. Searching Files

descriptorsList = os.listdir(descriptorDataFolder)
print ("This is your descriptors database: ")
print ("                                   ")
print descriptorsList

for descriptorFile in descriptorsList:    
    descriptorFilePath =(os.path.join(descriptorDataFolder,descriptorFile))
    print (descriptorFilePath)

    if descriptorUsed == "beats":
        if descriptorFile.endswith("_beats.csv"):
            with open(descriptorFilePath) as beats:
                beats1 = csv.reader(beats)
                for row in beats1:
                    dataBeats.append(float(row[1]))
                meanBeats = numpy.mean(dataBeats)
                #varBeats = numpy.var(dataBeats)
                dataBeats = []
                print meanBeats
                meanDataBeats.append(meanBeats)
                #varDataBeats.append(varBeats)

    if descriptorUsed == "attackstartendtimes":
        if descriptorFile.endswith("_attackstartendtimes.csv"):
            with open(descriptorFilePath) as attackstartendtimes:
                attackstartendtimes1 = csv.reader(attackstartendtimes)
                for row in attackstartendtimes1:
                    dataAttackStartEndTimes.append(float(row[1]))
                meanAttackStartEndTimes = numpy.mean(dataAttackStartEndTimes)
                #varAttackStartEndTimes = numpy.var(dataAttackStartEndTimes)
                dataAttackStartEndTimes = []
                print meanAttackStartEndTimes
                meanDataAttackStartEndTimes.append(meanAttackStartEndTimes)
                #varDataAttackStartEndTimes.append(varAttackStartEndTimes)

    if descriptorUsed == "logattacktime" or descriptorUsed == "All":
        while counter_logattacktime <= 0:
            headers.append("MeanLogattackTime")
            counter_logattacktime = counter_logattacktime + 1
        if descriptorFile.endswith("_logattacktime.csv"):
            with open(descriptorFilePath) as logattacktime:
                logattacktime1 = csv.reader(logattacktime)
                for row in logattacktime1:
                    dataLogAttackTime.append(float(row[1]))
                print dataLogAttackTime
                meanLogattacktime = numpy.mean(dataLogAttackTime)
                meanDataLogAttackTime.extend([meanLogattacktime])
                
                #varLogattacktime = numpy.var(dataLogAttackTime)
                #varDataLogAttackTime.extend([varLogattacktime])
                
                dataLogAttackTime = []
                print meanLogattacktime
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                print meanDataLogAttackTime
                print headers


    if descriptorUsed == "rms" or descriptorUsed == "All":
        while counter_rms <= 0:
            headers.append("MeanRms")
            counter_rms = counter_rms + 1
        if descriptorFile.endswith("_rms.csv"):
            with open(descriptorFilePath) as rms:
                rms1 = csv.reader(rms)
                for row in rms1:
                     dataRms.append(float(row[1]))
                print dataRms
                meanRms = numpy.mean(dataRms)
                meanDataRms.extend([meanRms])

                #varRms = numpy.var(dataRms)
                #varDataRms.extend([varRms])
                
                dataRms = []
                print meanRms
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                print meanDataRms
                print headers
        
    if descriptorUsed == "spectralcentroid" or descriptorUsed == "All":
        while counter_spectralcentroid <= 0:
            headers.append("MeanSpectralCentroid")
            counter_spectralcentroid = counter_spectralcentroid + 1
        if descriptorFile.endswith("_spectralcentroid.csv"):
            with open(descriptorFilePath) as spectralcentroid:
                spectralcentroid1 = csv.reader(spectralcentroid)
                for row in spectralcentroid1:
                     dataSpectralCentroid.append(float(row[1]))
                print dataSpectralCentroid
                meanSpectralCentroid = numpy.mean(dataSpectralCentroid)
                meanDataSpectralCentroid.extend([meanSpectralCentroid])

                #varSpectralCentroid = numpy.var(dataSpectralCentroid)
                #varDataSpectralCentroid.extend([varSpectralCentroid])
                
                dataSpectralCentroid = []
                print meanSpectralCentroid
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                print meanDataSpectralCentroid
                print headers

    if descriptorUsed == "spectralcrest" or descriptorUsed == "All":
        while counter_spectralcrest <= 0:
            headers.append("MeanSpectralCrest")
            counter_spectralcrest = counter_spectralcrest + 1
        if descriptorFile.endswith("_spectralcrest.csv"):
            with open(descriptorFilePath) as spectralcrest:
                spectralcrest1 = csv.reader(spectralcrest)
                for row in spectralcrest1:
                     dataSpectralCrest.append(float(row[1]))
                print dataSpectralCrest
                meanSpectralCrest = numpy.mean(dataSpectralCrest)
                meanDataSpectralCrest.extend([meanSpectralCrest])

                #varSpectralCrest = numpy.var(dataSpectralCrest)
                #varDataSpectralCrest.extend([varSpectralCrest])
                
                dataSpectralCrest = []
                print meanSpectralCrest
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanDataSpectralCrest
                print headers


    if descriptorUsed == "spectralflatness" or descriptorUsed == "All":
        while counter_spectralflatness <= 0:
                headers.append("MeanSpectralFlatness")
                counter_spectralflatness = counter_spectralflatness + 1
        if descriptorFile.endswith("_spectralflatness.csv"):
            with open(descriptorFilePath) as spectralflatness:
                spectralflatness1 = csv.reader(spectralflatness)
                for row in spectralflatness1:
                     dataSpectralFlatness.append(float(row[1]))
                print dataSpectralFlatness
                meanSpectralFlatness = numpy.mean(dataSpectralFlatness)
                meanDataSpectralFlatness.extend([meanSpectralFlatness])

                #varSpectralFlatness = numpy.var(dataSpectralFlatness)
                #varDataSpectralFlatness.extend([varSpectralFlatness])
                
                dataSpectralFlatness = []
                print meanSpectralFlatness
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanDataSpectralFlatness
                print headers

    if descriptorUsed == "spectralflux" or descriptorUsed == "All":
        while counter_spectralflux <= 0:
                headers.append("MeanSpectralFlux")
                counter_spectralflux = counter_spectralflux + 1
        if descriptorFile.endswith("_spectralflux.csv"):
            with open(descriptorFilePath) as spectralflux:
                spectralflux1 = csv.reader(spectralflux)
                for row in spectralflux1:
                     dataSpectralFlux.append(float(row[1]))
                print dataSpectralFlux
                meanSpectralFlux = numpy.mean(dataSpectralFlux)
                meanDataSpectralFlux.extend([meanSpectralFlux])

                #varSpectralFlux = numpy.var(dataSpectralFlux)
                #varDataSpectralFlux.extend([varSpectralFlux])
                
                dataSpectralFlux = []
                print meanSpectralFlux
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanDataSpectralFlux
                print headers

    if descriptorUsed == "spectralkurtosis" or descriptorUsed == "All":
        while counter_spectralkurtosis <= 0:
                headers.append("MeanSpectralKurtosis")
                counter_spectralkurtosis = counter_spectralkurtosis + 1
        if descriptorFile.endswith("_spectralkurtosis.csv"):
            with open(descriptorFilePath) as spectralkurtosis:
                spectralkurtosis1 = csv.reader(spectralkurtosis)
                for row in spectralkurtosis1:
                     dataSpectralKurtosis.append(float(row[1]))
                print dataSpectralKurtosis
                meanSpectralKurtosis = numpy.mean(dataSpectralKurtosis)
                meanDataSpectralKurtosis.extend([meanSpectralKurtosis])

                #varSpectralKurtosis = numpy.var(dataSpectralKurtosis)
                #varDataSpectralKurtosis.extend([varSpectralKurtosis])
                
                dataSpectralKurtosis = []
                print meanSpectralKurtosis
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanDataSpectralKurtosis
                print headers

    if descriptorUsed == "spectralrolloff" or descriptorUsed == "All":
        while counter_spectralrolloff <= 0:
                headers.append("MeanSpectralRollOff")
                counter_spectralrolloff = counter_spectralrolloff + 1
        if descriptorFile.endswith("_spectralrolloff.csv"):
            with open(descriptorFilePath) as spectralrolloff:
                spectralrolloff1 = csv.reader(spectralrolloff)
                for row in spectralrolloff1:
                     dataSpectralRollOff.append(float(row[1]))
                print dataSpectralRollOff
                meanSpectralRollOff = numpy.mean(dataSpectralRollOff)
                meanDataSpectralRollOff.extend([meanSpectralRollOff])

                #varSpectralRollOff = numpy.var(dataSpectralRollOff)
                #varDataSpectralRollOff.extend([varSpectralRollOff])
                
                dataSpectralRollOff = []
                print meanSpectralRollOff
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanDataSpectralRollOff
                print headers

    if descriptorUsed == "spectralskewness" or descriptorUsed == "All":
        while counter_spectralskewness <= 0:
                headers.append("MeanSpectralSkewness")
                counter_spectralskewness = counter_spectralskewness + 1
        if descriptorFile.endswith("_spectralskewness.csv"):
            with open(descriptorFilePath) as spectralskewness:
                spectralskewness1 = csv.reader(spectralskewness)
                for row in spectralskewness1:
                     dataSpectralSkewness.append(float(row[1]))
                print dataSpectralSkewness
                meanSpectralSkewness = numpy.mean(dataSpectralSkewness)
                meanDataSpectralSkewness.extend([meanSpectralSkewness])

                #varSpectralSkewness = numpy.var(dataSpectralSkewness)
                #varDataSpectralSkewness.extend([varSpectralSkewness])
                
                dataSpectralSkewness = []
                print meanSpectralSkewness
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanDataSpectralSkewness
                print headers

    if descriptorUsed == "spectralspread" or descriptorUsed == "All":
        while counter_spectralspread <= 0:
                headers.append("MeanSpectralSpread")
                counter_spectralspread = counter_spectralspread + 1
        if descriptorFile.endswith("_spectralspread.csv"):
            with open(descriptorFilePath) as spectralspread:
                spectralspread1 = csv.reader(spectralspread)
                for row in spectralspread1:
                     dataSpectralSpread.append(float(row[1]))
                print dataSpectralSpread
                meanSpectralSpread = numpy.mean(dataSpectralSpread)
                meanDataSpectralSpread.extend([meanSpectralSpread])

                #varSpectralSpread = numpy.var(dataSpectralSpread)
                #varDataSpectralSpread.extend([varSpectralSpread])
                
                dataSpectralSpread = []
                print meanSpectralSpread
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanDataSpectralSpread
                print headers

    if descriptorUsed == "temporalcentroid" or descriptorUsed == "All":
        while counter_temporalcentroid <= 0:
                headers.append("MeanTemporalCentroid")
                counter_temporalcentroid = counter_temporalcentroid + 1
        if descriptorFile.endswith("_temporalcentroid.csv"):
            with open(descriptorFilePath) as temporalcentroid:
                temporalcentroid1 = csv.reader(temporalcentroid)
                for row in temporalcentroid1:
                     dataTemporalCentroid.append(float(row[0]))
                print dataTemporalCentroid
                meanTemporalCentroid = numpy.mean(dataTemporalCentroid)
                meanDataTemporalCentroid.extend([meanTemporalCentroid])

                #varTemporalCentroid = numpy.var(dataTemporalCentroid)
                #varDataTemporalCentroid.extend([varTemporalCentroid])

                
                dataTemporalCentroid = []
                print meanTemporalCentroid
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanTemporalCentroid
                print headers

    if descriptorUsed == "zerocrossingrate" or descriptorUsed == "All":
        while counter_zerocrossingrate <= 0:
                headers.append("MeanZeroCrossingRate")
                counter_zerocrossingrate = counter_zerocrossingrate + 1
        if descriptorFile.endswith("_zerocrossingrate.csv"):
            with open(descriptorFilePath) as zerocrossingrate:
                zerocrossingrate1 = csv.reader(zerocrossingrate)
                for row in zerocrossingrate1:
                     dataZeroCrossingRate.append(float(row[1]))
                print dataZeroCrossingRate
                meanZeroCrossingRate = numpy.mean(dataZeroCrossingRate)
                meanDataZeroCrossingRate.extend([meanZeroCrossingRate])

                #varZeroCrossingRate = numpy.var(dataZeroCrossingRate)
                #varDataZeroCrossingRate.extend([varZeroCrossingRate])
                
                dataZeroCrossingRate = []
                print meanZeroCrossingRate
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanZeroCrossingRate
                print headers

    if descriptorUsed == "mfcc" or descriptorUsed == "All":
        while counter_mfcc <= 0:
                headers.append("MeanMfcc0")
                headers.extend(["MeanMfcc1","MeanMfcc2","MeanMfcc3","MeanMfcc4","MeanMfcc5","MeanMfcc6","MeanMfcc7","MeanMfcc8","MeanMfcc9","MeanMfcc10","MeanMfcc11","MeanMfcc12","MeanMfcc13","MeanMfcc14","MeanMfcc15","MeanMfcc16","MeanMfcc17","MeanMfcc18","MeanMfcc19"])
                counter_mfcc = counter_mfcc + 1
        if descriptorFile.endswith("mfcc_coefficients.csv"):
            with open(descriptorFilePath) as mfcc:
                mfcc1 = csv.reader(mfcc)
                for row in mfcc1:
                     dataMfcc0.append(float(row[1]))
                     dataMfcc1.append(float(row[2]))
                     dataMfcc2.append(float(row[3]))
                     dataMfcc3.append(float(row[4]))
                     dataMfcc4.append(float(row[5]))
                     dataMfcc5.append(float(row[6]))
                     dataMfcc6.append(float(row[7]))
                     dataMfcc7.append(float(row[8]))
                     dataMfcc8.append(float(row[9]))
                     dataMfcc9.append(float(row[10]))
                     dataMfcc10.append(float(row[11]))
                     dataMfcc11.append(float(row[12]))
                     dataMfcc12.append(float(row[13]))
                     dataMfcc13.append(float(row[14]))
                     dataMfcc14.append(float(row[15]))
                     dataMfcc15.append(float(row[16]))
                     dataMfcc16.append(float(row[17]))
                     dataMfcc17.append(float(row[18]))
                     dataMfcc18.append(float(row[19]))
                     dataMfcc19.append(float(row[20]))
                print dataMfcc0,dataMfcc1,dataMfcc2,dataMfcc3,dataMfcc4,dataMfcc5,dataMfcc6,dataMfcc7,dataMfcc8,dataMfcc9,dataMfcc10,dataMfcc11,dataMfcc12,dataMfcc13,dataMfcc14,dataMfcc15,dataMfcc16,dataMfcc17,dataMfcc18,dataMfcc19
                meanMfcc0 = numpy.mean(dataMfcc0)
                meanDataMfcc0.extend([meanMfcc0])
                meanMfcc1 = numpy.mean(dataMfcc1)
                meanDataMfcc1.extend([meanMfcc1])
                meanMfcc2 = numpy.mean(dataMfcc2)
                meanDataMfcc2.extend([meanMfcc2])
                meanMfcc3 = numpy.mean(dataMfcc3)
                meanDataMfcc3.extend([meanMfcc3])
                meanMfcc4 = numpy.mean(dataMfcc4)
                meanDataMfcc4.extend([meanMfcc4])
                meanMfcc5 = numpy.mean(dataMfcc5)
                meanDataMfcc5.extend([meanMfcc5])
                meanMfcc6 = numpy.mean(dataMfcc6)
                meanDataMfcc6.extend([meanMfcc6])
                meanMfcc7 = numpy.mean(dataMfcc7)
                meanDataMfcc7.extend([meanMfcc7])
                meanMfcc8 = numpy.mean(dataMfcc8)
                meanDataMfcc8.extend([meanMfcc8])
                meanMfcc9 = numpy.mean(dataMfcc9)
                meanDataMfcc9.extend([meanMfcc9])
                meanMfcc10 = numpy.mean(dataMfcc10)
                meanDataMfcc10.extend([meanMfcc10])
                meanMfcc11 = numpy.mean(dataMfcc11)
                meanDataMfcc11.extend([meanMfcc11])
                meanMfcc12 = numpy.mean(dataMfcc12)
                meanDataMfcc12.extend([meanMfcc12])
                meanMfcc13 = numpy.mean(dataMfcc13)
                meanDataMfcc13.extend([meanMfcc13])
                meanMfcc14 = numpy.mean(dataMfcc14)
                meanDataMfcc14.extend([meanMfcc14])
                meanMfcc15 = numpy.mean(dataMfcc15)
                meanDataMfcc15.extend([meanMfcc15])
                meanMfcc16 = numpy.mean(dataMfcc16)
                meanDataMfcc16.extend([meanMfcc16])
                meanMfcc17 = numpy.mean(dataMfcc17)
                meanDataMfcc17.extend([meanMfcc17])
                meanMfcc18 = numpy.mean(dataMfcc18)
                meanDataMfcc18.extend([meanMfcc18])
                meanMfcc19 = numpy.mean(dataMfcc19)
                meanDataMfcc19.extend([meanMfcc19])
                dataMfcc0 = []
                dataMfcc1 = []
                dataMfcc2 = []
                dataMfcc3 = []
                dataMfcc4 = []
                dataMfcc5 = []
                dataMfcc6 = []
                dataMfcc7 = []
                dataMfcc8 = []
                dataMfcc9 = []
                dataMfcc10 = []
                dataMfcc11 = []
                dataMfcc12 = []
                dataMfcc13 = []
                dataMfcc14 = []
                dataMfcc15 = []
                dataMfcc16 = []
                dataMfcc17 = []
                dataMfcc18 = []
                dataMfcc19 = []
                
                #print meanMfcc
                if descriptorDataFolder.endswith("blues"):
                    genre.append("blu")
                if descriptorDataFolder.endswith("classical"):
                    genre.append("cla")
                if descriptorDataFolder.endswith("country"):
                    genre.append("count")
                if descriptorDataFolder.endswith("disco"):
                    genre.append("disco")
                if descriptorDataFolder.endswith("hiphop"):
                    genre.append("hip")
                if descriptorDataFolder.endswith("jazz"):
                    genre.append("jazz")
                if descriptorDataFolder.endswith("metal"):
                    genre.append("met")
                if descriptorDataFolder.endswith("pop"):
                    genre.append("pop")
                if descriptorDataFolder.endswith("reggae"):
                    genre.append("reg")
                if descriptorDataFolder.endswith("rock"):
                    genre.append("rock")
                #print meanMfcc
                print headers


headers.extend(["Genre"])
totalData.extend([meanDataLogAttackTime,meanDataRms,meanDataSpectralCentroid,meanDataSpectralCrest,meanDataSpectralFlatness,meanDataSpectralFlux,meanDataSpectralKurtosis,meanDataSpectralRollOff,meanDataSpectralSkewness,meanDataSpectralSpread,meanDataTemporalCentroid,meanDataZeroCrossingRate,meanDataMfcc0,meanDataMfcc1,meanDataMfcc2,meanDataMfcc3,meanDataMfcc4,meanDataMfcc5,meanDataMfcc6,meanDataMfcc7,meanDataMfcc8,meanDataMfcc9,meanDataMfcc10,meanDataMfcc11,meanDataMfcc12,meanDataMfcc13,meanDataMfcc14,meanDataMfcc15,meanDataMfcc16,meanDataMfcc17,meanDataMfcc18,meanDataMfcc19,genre])
x = zip(*totalData)
print x

#print totalData
if descriptorDataFolder.endswith("blues"):
    with open(statisticsDataSaving+"\\"+"Statistics_blues.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(x)
if descriptorDataFolder.endswith("classical"):
    with open(statisticsDataSaving+"\\"+"Statistics_classical.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(x)
if descriptorDataFolder.endswith("country"):
    with open(statisticsDataSaving+"\\"+"Statistics_country.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(x)
if descriptorDataFolder.endswith("disco"):
    with open(statisticsDataSaving+"\\"+"Statistics_disco.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(x)
if descriptorDataFolder.endswith("hiphop"):
    with open(statisticsDataSaving+"\\"+"Statistics_hiphop.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(x)
if descriptorDataFolder.endswith("jazz"):
    with open(statisticsDataSaving+"\\"+"Statistics_jazz.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)# + "varDataLogAttackTime,varDataRms,varnDataSpectralCentroid,varDataSpectralCrest,varDataSpectralFlatness,varDataSpectralFlux,varDataSpectralKurtosis,varDataSpectralRollOff,varDataSpectralSkewness,varDataSpectralSpread,varDataTemporalCentroid,varDataZeroCrossingRate")
        writer.writerows(x)
if descriptorDataFolder.endswith("metal"):
    with open(statisticsDataSaving+"\\"+"Statistics_metal.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)# + "varDataLogAttackTime,varDataRms,varnDataSpectralCentroid,varDataSpectralCrest,varDataSpectralFlatness,varDataSpectralFlux,varDataSpectralKurtosis,varDataSpectralRollOff,varDataSpectralSkewness,varDataSpectralSpread,varDataTemporalCentroid,varDataZeroCrossingRate")
        writer.writerows(x)
if descriptorDataFolder.endswith("pop"):
    with open(statisticsDataSaving+"\\"+"Statistics_pop.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)# + "varDataLogAttackTime,varDataRms,varnDataSpectralCentroid,varDataSpectralCrest,varDataSpectralFlatness,varDataSpectralFlux,varDataSpectralKurtosis,varDataSpectralRollOff,varDataSpectralSkewness,varDataSpectralSpread,varDataTemporalCentroid,varDataZeroCrossingRate")
        writer.writerows(x)
if descriptorDataFolder.endswith("reggae"):
    with open(statisticsDataSaving+"\\"+"Statistics_reggae.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)# + "varDataLogAttackTime,varDataRms,varnDataSpectralCentroid,varDataSpectralCrest,varDataSpectralFlatness,varDataSpectralFlux,varDataSpectralKurtosis,varDataSpectralRollOff,varDataSpectralSkewness,varDataSpectralSpread,varDataTemporalCentroid,varDataZeroCrossingRate")
        writer.writerows(x)
if descriptorDataFolder.endswith("rock"):
    with open(statisticsDataSaving+"\\"+"Statistics_rock.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)# + "varDataLogAttackTime,varDataRms,varnDataSpectralCentroid,varDataSpectralCrest,varDataSpectralFlatness,varDataSpectralFlux,varDataSpectralKurtosis,varDataSpectralRollOff,varDataSpectralSkewness,varDataSpectralSpread,varDataTemporalCentroid,varDataZeroCrossingRate")
        writer.writerows(x)

print ("")
print ("Computing Done")
    
