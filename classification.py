from loadDataset import readFromCSV
from sklearn.ensemble import RandomForestClassifier
from time import time
from data import Dataset
from saveResults import saveResults
for i in range(0,5,1):
    dataset = readFromCSV('\t')
    train_data = Dataset(dataset[0])
    test_data = Dataset(dataset[1])
    classifier = RandomForestClassifier()
    time0 = time()
    classifier.fit(train_data.get_features(), train_data.get_classes())
    classifierName = type(classifier).__name__
    accuracy = (classifier.score(test_data.get_features(), test_data.get_classes()))
    nIter = '---'
    # nIter = (classifier.n_iter_)
    print "Accuracy: ", accuracy
    saveResults(classifierName,accuracy,train_data.size[0],test_data.size[0],time()-time0,nIter)