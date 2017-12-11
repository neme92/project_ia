class dysgrDataClass():

    def __init__(self, imageName, isWithoutDysgraphia, diffAngle, performanceTime):
        self.imageName = imageName
        self.isWithoutDysgraphia = isWithoutDysgraphia 
        self.diffAngle = diffAngle 
        self.performanceTime = performanceTime 

def loadFeatures():
    dbList = []

    #retrieve from csv (in this case .txt) every entry 
    fileTxt = open("db.txt", "r")
    for line in fileTxt:
        if(not line.startswith("#")):
            splittedLine = line.split(',')
            #print("Feature database: Entering " + splittedLine[0], splittedLine[1], splittedLine[2], splittedLine[3].replace("\n", ""))
            listEntry = dysgrDataClass(splittedLine[0], splittedLine[1], splittedLine[2], splittedLine[3])
            dbList.append(listEntry)

    if not dbList:
        print("There was a problem while loading features from db. Aborting now")
        sys.exit(0)
    else:
        print("dbList has " + str(len(dbList)) + " entries")

    fileTxt.close()
    return dbList