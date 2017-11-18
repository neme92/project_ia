import urllib, re, random, os, math, sys
from PIL import Image
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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




''' ----------------------------------------------------------------------------------------- '''

imgDirectory = os.getcwd() + "/img/"
#image folder destination
checkDownloadFolder(imgDirectory)   

#creating an empty array which has to be fullfilled with words from words.txt file
arrayWord = []
with open('10k.txt','r') as f:
    for line in f:
        for word in line.split():
           arrayWord.append(word)   

#        prototype of the request:
# https://www.cs.toronto.edu/~graves/handwriting.cgi?text=texthere&style=&bias=0.9&samples=5
# var 
#   text=texthere       string which has to be written
#   style=              if empty, it should be random
#   bias=0.9            [0,1] 1 digit after comma only
#   samples=5           [1,5]

with open("db.txt", "w") as resultFile:
    #counter variable is used just to print how many requestes have been done
    counter = 0

    resultFile.write("imageName, isWithoutDysgraphia,diffAngle, performanceTime \n\n")
    #iterate the arrayWord
    for word in range(len(arrayWord)):

        refHeight = 0
        refWidth = 0
        refAngle = 0
        text = arrayWord[word]                    #retrieve the word from array

        samples = 5

        for isWithoutDysgraphia in xrange(1, -1, -1):

            bias=str(isWithoutDysgraphia)                         #choosing the bias with values between 0 or 1 
            #samples = 1 if isWithoutDysgraphia else 2;           #this gives 1 sample if isWithoutDysgraphia, 2 samples if lowQuality

            #print on screen status of requests
            printTrack(str(counter), str(len(arrayWord)), str(samples), text)

            #compose the url
            url = "https://www.cs.toronto.edu/~graves/handwriting.cgi?text=" + text + "&style=&bias=" + bias + "&samples=" + str(samples)

            html = getHtmlFromURL(url)

            imgs = extractImages(html)

            #this array holds the 2-tuples of reference images' dimensions, needed to compute an average value
            referenceArray = list()          

            #retrieve and save into "/img" folder
            for i in range(len(imgs)):
                imageName = text + "_" + str(bias) + "_[" + str(i) + "].png"
                imagePath = os.path.join(imgDirectory, imageName)
                urllib.urlretrieve("data:image/" + imgs[i], imagePath)

                imgFile = Image.open(imagePath)

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
                if imgFile.mode != 'RGB':
                    imgFile = imgFile.convert('RGB')
                imgFile.save(imagePath)

                perfTime = getStatisticalValues(isWithoutDysgraphia)
                resultFile.write(imageName + "," + str(bias) + "," + str(diffAngle) + "," + str(perfTime) + "\n")

            counter+= 1
f.close()