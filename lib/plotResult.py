import matplotlib.pyplot as plt 

def plot(trainingResults, testResults):
    # TRAINING
    #   ---   LOSS
    plt.plot([row[1] for row in trainingResults])
    plt.title('Training')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()

    #   ---   ACCURACY
    plt.plot([row[2] for row in trainingResults])
    plt.title('Training')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.show()

    # TESTING
    #   ---   LOSS
    plt.plot([row[1] for row in testResults])
    plt.title('Testing')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.show()

    #   ---   ACCURACY
    plt.plot([row[2] for row in testResults])
    plt.title('Testing')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.show()
