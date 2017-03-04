import os
def saveResults(classifierName, accuracy, numOfTrainEx, numOfTestEx, convTime, nIter):
    filepaths = []
    with open("/Users/apple/PycharmProjects/Breast-Cancer-Classification/filepaths") as f:
        for line in f:
            filepaths.append(line.replace('\n', ''))
    with open(filepaths[2],'a') as resultFile:
        if os.path.getsize(filepaths[2]) == 0:
            heading = "Classifier name,Accuracy,Num of training examples,Num of test examples, Convergence time, Num of iterations\n"
            resultFile.write(heading)
        results = str(classifierName)+","+str(accuracy)+","+str(numOfTrainEx)+","+str(numOfTestEx)+","+str(convTime)+","+str(nIter)+'\n'
        resultFile.write(results)