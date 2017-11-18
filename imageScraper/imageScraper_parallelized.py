import urllib, re, random, os, math, sys, threading, multiprocessing
from PIL import Image
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool

counter = 0
numberOfWords = 0

#check if destination folder exists, if not create it
def checkDownloadFolder(imgDirectory):
    if not os.path.exists(imgDirectory):
        os.makedirs(imgDirectory)

def getAngle(shorterCathetus, longerCathetus):
    return np.arctan(float(shorterCathetus)/float(longerCathetus))

#distribution data taken from paper "Handwriting Performance, Self-Reports, and Perceived Self-Efficacy Among Children With Dysgraphia"
#table page 188/7 using "Computerized Penmanship Evaluation Tool"
def getTimePerformance(isWithoutDysgraphia):
    lowBound, upBound = 0, 5                        #taken from hpsq questionnaire
    mean, standardDeviation = 2.36, 0.45            #case without Dysgraphia (since more samples are required, it is here as default value)    

    if isWithoutDysgraphia is 1:                    #case children With Dysgraphia
        mean, standardDeviation = 1.00, 0.66        #changes mean and DS values
    
    return np.random.normal(mean, standardDeviation, None)
    
def getStatisticalValues(isWithoutDysgraphia):
    #for future works could take into account: TotalTime, onPaperTime, inAirTime, meanPresure, meanWritingVelocity 
    #it now calculates TotalTimeOnly
    while True:
        res = round(getTimePerformance(isWithoutDysgraphia), 3)
        if res > 0.20:          
            #this IF protects results from Flash writers or results
            break
    return res

def printTrack(wordNumber, rangeArray, numberOfSamples, text):
    #this just adds a fancy animation
    animationSet = "|/-\\"
    animChar = animationSet[int(wordNumber) % len(animationSet)]
    #end of eyecandy effect

    sys.stdout.write("\r" + animChar + "  " + wordNumber + " of " + rangeArray + ": getting " + numberOfSamples + " img(s) for word '" + text + "'" + " " * 20)
    sys.stdout.flush()

#input: image output: binarized image
def binarizeImage(image):
    return  image.convert('1')

#scales input image to imageSize
def scaleImage(image):
    maxWidth, maxHeight =  512, 128
    imageSize = maxWidth, maxHeight
    whiteCanvas = Image.new("RGB", imageSize, "white")
    imgTmp = image
    
    #this scales only if it is bigger than imageSize
    imgTmp.thumbnail(imageSize, Image.ANTIALIAS)  

    whiteCanvas.paste(imgTmp, (0, (maxHeight - imgTmp.height)))

    return whiteCanvas   


#This takes the image as input and gives as result the image formatted:
#A formatted image is before scaled to a fixed dimension and subsequently binarized
def formatImage(isWithoutDysgraphia, img):
    imgTmp = scaleImage(img)                
    imgTmp = binarizeImage(imgTmp)          
    
    return imgTmp

def getHtmlFromURL(url):
    website = urllib.urlopen(url)
    return website.read()

def extractImages(html):
    #looks for the img elements which starts with "data:image" string: 
    #this because the images we are interested to get have this string at the beginning
    pat = re.compile (r'<img [^>]*src="data:image/([^"]+)')
    return pat.findall(html)

def computeReferenceDimensions(referenceArray):
    refHeight, refWidth, refAngle = 0, 0, 0

    for dim in referenceArray:
        H, W = dim
        refHeight = refHeight + H
        refWidth = refWidth + W

    refHeight = refHeight / len(referenceArray)
    refWidth = refWidth / len(referenceArray)

    refAngle = getAngle(refHeight, refWidth)

    #print("Average refHeight: " + str(refHeight) + ", refWidth " + str(refWidth) + ", Angle " + str(refAngle))

    return refHeight, refWidth, refAngle


def worker(word):
    
    #resulting string which has to be written in the file
    resString = ""

    #ref variables used to compare High with Low quality images
    refHeight = 0
    refWidth = 0
    refAngle = 0

    #var of the request                   
    text = word                                 #retrieve the word from array
    samples = 5

    #two requests:
    #   1 for High Quality text (first one, which fixes ref variables)
    #   1 for Low Quality text  (second one)
    for isWithoutDysgraphia in xrange(1, -1, -1):

        bias=str(isWithoutDysgraphia)                   #choosing the bias with values between 0 or 1 

        #compose the url
        url = "https://www.cs.toronto.edu/~graves/handwriting.cgi?text=" + text + "&style=&bias=" + bias + "&samples=" + str(samples)

        #request page
        html = getHtmlFromURL(url)

        #array holding images downloaded from page
        imgs = extractImages(html)

        #this array holds the 2-tuples of ref images' dimensions, needed to compute an average value
        referenceArray = list()          

        #retrieve and save into "/img" folder
        for i in range(len(imgs)):
            imageName = text + "_" + str(bias) + "_[" + str(i) + "].png"
            imagePath = os.path.join(imgDirectory, imageName)
            urllib.urlretrieve("data:image/" + imgs[i], imagePath)

            imgFile = Image.open(imagePath)

            #starting computing angle
            diffAngle = 0

            if(isWithoutDysgraphia):
                referenceArray.append(imgFile.size)              #saves reference height and width

                if(i is samples-1):
                    refHeight, refWidth, refAngle = computeReferenceDimensions(referenceArray)
                    #print("\nrefWidth: "  + str(refWidth) + ", refHeight: " + str(refHeight) + ", Angle of ref img is " + str(refAngle))
            else:
                #get the diagonal angle after the image was stretched referring to the highQuality witdh: after that calculate the angles (highQ and lowQ) and save the slope as difference between them in the name or cvs
                lowQHeight, lowQWidth = imgFile.size            #save height and width
                scaleFactor = float(refWidth) / float(lowQWidth)
                diffAngle = round(math.degrees(np.absolute(refAngle - getAngle(lowQHeight * scaleFactor, lowQWidth * scaleFactor)))%360, 3)
                #print("Scale factor: " + str(scaleFactor))    
                #print("\nlowQWidth: "  + str(lowQWidth) + ", lowQHeight: " + str(lowQHeight) + ", Angle of lowQ img is " + str(diffAngle))
            
            imgFile = formatImage(isWithoutDysgraphia, imgFile)

            #needed to correctly print the image
            if imgFile.mode != 'RGB':
                imgFile = imgFile.convert('RGB')
            imgFile.save(imagePath)

            #computing the performance time thanks to statistical data taken from paper
            perfTime = getStatisticalValues(isWithoutDysgraphia)

            # in order to write all lines of a single word together, 
            # here a tmp string is build and then written in the file once it's completed: 
            # this reduces locks on file and makes the file eaasier to read
            resString += imageName + "," + str(bias) + "," + str(diffAngle) + "," + str(perfTime) + "\n"

    writeOnFile(resString, word)

#   Function writeOnFile has lock because it will write on a shared file
#   in order to avoid interrupt and append in a sync way

lock = threading.Lock()

def writeOnFile(fileStr, word):
    lock.acquire()

    if(word != ""):
        global counter
        global numberOfWords

        counter = counter + 1
        print (str(counter/numberOfWords) + "% " + " - Thread " + str((multiprocessing.current_process().pid)%4) + " working on '" + word + "'")

    try:
        with open("db.txt", "a") as resultFile: 
            resultFile.write(fileStr)
        f.close()
    finally:
        lock.release()

def printSign():
    print """
        \t____   ____.__      __________.__        
        \t\   \ /   /|__| ____\______   \__| ____  
        \t \   V   / |  |/    \|       _/  |/    \ 
        \t  \     /  |  |   |  \    |   \  |   |  \ 
        \t   \___/   |__|___|  /____|_  /__|___|  /
        \t                   \/       \/        \/ 
    Automatic generator of hadwritten images with stats\n\n"""

''' ----------------------------------------------------------------------------------------- '''

printSign()

''' list of provided dictionaries
    uncomment the one you want to use   '''

#filename = 'shortWords.txt'
#filename = 'words_LP.txt'
filename = '10k.txt'
#filename = '20k.txt'

print("Reading from file: '" + filename + "'\n\n\n")

imgDirectory = os.getcwd() + "/img/"
#image folder destination
checkDownloadFolder(imgDirectory)   

#creating an empty array which has to be fullfilled with words from words.txt file
arrayWord = []
with open(filename,'r') as f:
    for line in f:
        for word in line.split():
           arrayWord.append(word)   


#        prototype of the request:
# https://www.cs.toronto.edu/~graves/handwriting.cgi?text=texthere&style=&bias=0.9&samples=5
# vars: 
#   text=texthere       string which has to be written
#   style=              if empty, it should be random
#   bias=0.9            [0,1] 1 digit after comma only
#   samples=5           [1,5]

#heading file, can be omitted
writeOnFile("imageName, isWithoutDysgraphia,diffAngle, performanceTime \n", "")

#instantiating the threadpool and starting it
THREAD_NUMBER = 4

global numberOfWords
numberOfWords = len(arrayWord)/THREAD_NUMBER

p = multiprocessing.Pool(THREAD_NUMBER)
p.map(worker, arrayWord)

print("\n\nThat was a huge amout of words, my job here is done")