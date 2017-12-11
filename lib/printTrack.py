import sys
globalCounter = 0

def printTrack():
    global globalCounter
    #this just adds a fancy animation
    animationSet = "|/-\\"
    animChar = animationSet[globalCounter % len(animationSet)]
    #end of eyecandy effect

    sys.stdout.write("\r" + animChar + "  Forwarding ")
    sys.stdout.flush()
    globalCounter += 1