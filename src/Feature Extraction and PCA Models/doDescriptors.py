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
print ("---------------------------------Info-------------------------------------")
print ("This is a script that calculates Audio Descriptors using Sonic Annotator")
print ("and Vamp Plugins developed by Queens Mary and UPF University.")
print ("                                                            ")
print ("David Campayo, Final Degree Project, Audiovisual Systems Engineering, 2018")
print ("--------------------------------------------------------------------------")
print ("                                                            ")

#READING COMMAND LINE

if len(sys.argv) < 4:
    print ("----------------------------- How to use it-------------------------------")
    print ("                                                                 ")
    print ("Declaration:            -> doDescriptors.py path\\to\\audio\\folder path\\to\\data\\saving descriptor")
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
audioDataFolder = sys.argv[1]
audioDataSaving = sys.argv[2]
descriptorUsed = sys.argv[3]

#DECLATION COUNTERS

counter_beats = 0
counter_attackstartendtimes = 0
counter_logattack = 0
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

audioList = os.listdir(audioDataFolder)
print ("This is your database: ")
print ("                       ")

for audioFile in audioList:
    print (audioFile)    
    audioFilePath =(os.path.join(audioDataFolder,audioFile))
    print (audioFilePath)
    if audioFile.endswith(".mp3"):# Change to MP3  OR  WAV when it is required 
        if descriptorUsed == "beats" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:beatroot-vamp:beatroot:beats > " + audioFile + "_beats.n3")
            os.system("sonic-annotator -t " + audioFile + "_beats.n3 "+ audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_beats.n3")
            
        if descriptorUsed == "attackstartendtimes" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:attackstartendtimes:attackstartendtimes > " + audioFile + "_attackstartendtimes.n3")
            os.system("sonic-annotator -t " + audioFile + "_attackstartendtimes.n3 "+ audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_attackstartendtimes.n3")
            
        if descriptorUsed == "logattacktime" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:logattacktime:logattacktime > " + audioFile + "_logattack.n3")
            os.system("sonic-annotator -t " + audioFile + "_logattack.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_logattack.n3")
            
        if descriptorUsed == "rms" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:rms:rms > " + audioFile + "_rms.n3")
            os.system("sonic-annotator -t " + audioFile + "_rms.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_rms.n3")
                    
        if descriptorUsed == "spectralcentroid" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:spectralcentroid:spectralcentroid > " + audioFile + "_spectralcentroid.n3")
            os.system("sonic-annotator -t " + audioFile + "_spectralcentroid.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_spectralcentroid.n3")
            
        if descriptorUsed == "spectralcrest" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:spectralcrest:spectralcrest > " + audioFile + "_spectralcrest.n3")
            os.system("sonic-annotator -t " + audioFile + "_spectralcrest.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_spectralcrest.n3")

        if descriptorUsed == "spectralflatness" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:spectralflatness:spectralflatness > " + audioFile + "_spectralflatness.n3")
            os.system("sonic-annotator -t " + audioFile + "_spectralflatness.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)          
            os.remove(audioFile + "_spectralflatness.n3")
            
        if descriptorUsed == "spectralflux" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:spectralflux:spectralflux > " + audioFile + "_spectralflux.n3")
            os.system("sonic-annotator -t " + audioFile + "_spectralflux.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_spectralflux.n3")
            
        if descriptorUsed == "spectralkurtosis" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:spectralkurtosis:spectralkurtosis > " + audioFile + "_spectralkurtosis.n3")
            os.system("sonic-annotator -t " + audioFile + "_spectralkurtosis.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_spectralkurtosis.n3")

            
        if descriptorUsed == "spectralrolloff" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:spectralrolloff:spectralrolloff > " + audioFile + "_spectralrolloff.n3")
            os.system("sonic-annotator -t " + audioFile + "_spectralrolloff.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_spectralrolloff.n3")
            
        if descriptorUsed == "spectralskewness" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:spectralskewness:spectralskewness > " + audioFile + "_spectralskewness.n3")
            os.system("sonic-annotator -t " + audioFile + "_spectralskewness.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_spectralskewness.n3")

        if descriptorUsed == "spectralspread" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:spectralspread:spectralspread > " + audioFile + "_spectralspread.n3")
            os.system("sonic-annotator -t " + audioFile + "_spectralspread.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_spectralspread.n3")

        if descriptorUsed == "temporalcentroid" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:temporalcentroid:temporalcentroid > " + audioFile + "_temporalcentroid.n3")
            os.system("sonic-annotator -t " + audioFile + "_temporalcentroid.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_temporalcentroid.n3")

        if descriptorUsed == "zerocrossingrate" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:mir-edu:zerocrossingrate:zerocrossingrate > " + audioFile + "_zerocrossingrate.n3")
            os.system("sonic-annotator -t " + audioFile + "_zerocrossingrate.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_zerocrossingrate.n3")

        if descriptorUsed == "mfcc" or descriptorUsed == "All":
            print ("Computing " + descriptorUsed + "for " + audioFile)
            os.system(sonic_annotator + "\\" + "sonic-annotator -s vamp:qm-vamp-plugins:qm-mfcc:coefficients > " + audioFile + "_mfcc.n3")
            os.system("sonic-annotator -t " + audioFile + "_mfcc.n3 " + audioFilePath + " -w csv --csv-basedir " + audioDataSaving)
            os.remove(audioFile + "_mfcc.n3")
            
print ("")
print ("Computing Done")
    
