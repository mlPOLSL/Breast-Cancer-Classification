from loadDataset import readFromCSV
from sklearn.neural_network import MLPClassifier
from data import Dataset
dataset = readFromCSV('\t')
train_data = Dataset(dataset[0])
test_data = Dataset(dataset[1])
classifier = MLPClassifier((20, 10), learning_rate_init=0.001, max_iter=1000 )
classifier.fit(train_data.get_features(), train_data.get_classes())
print(classifier.score(test_data.get_features(), test_data.get_classes()))
print(classgiifier.n_iter_)