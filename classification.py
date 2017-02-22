from loadDataset import readFromCSV
from sklearn.neural_network import MLPClassifier

dataset = readFromCSV('\t')

classifier = MLPClassifier((20, 10), learning_rate_init=0.04, momentum=0.4)
classifier.fit(dataset[0][:, 0:8], dataset[0][:, 9])
print classifier.score(dataset[1][:, 0:8], dataset[1][:, 9])
print classifier.n_iter_