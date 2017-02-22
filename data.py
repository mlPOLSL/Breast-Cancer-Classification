from numpy import divide

class Dataset:
    def __init__(self, data):
        self.data = data
        self.size = [data.shape[0],data.shape[1]]
    def get_features(self):
        return self.data[:,0:self.size[1]-1]

    def get_classes(self):
        return self.data[:,self.size[1]-1]

    def normalize(self, method="standarization"):
        if method == "rescaling":
            #Rescaling method
            max = self.data.max(axis=0)
            min = self.data.min(axis=0)
            for i in range(0,self.size[1]-1):
                self.data[:,i] = (self.data[:,i] - min[i]) / (max[i] - min[i])
        else:
            # Standarization method
            mean = self.data.mean(axis=0)
            denominator = self.data.std(axis=0)
            for i in range(0, self.size[1] - 1):
                self.data[:, i] = divide((self.data[:, i] - mean[i]), denominator[i])

class Normalizator():
    def __init__(self,dataset,method="standarization"):
        self.train_data = dataset.data
        self.size = self.train_data.shape
        self.mean = self.train_data.mean(axis=0)
        self.std = self.train_data.std(axis = 0)
        self.max = self.train_data.max(axis = 0)
        self.min = self.train_data.min(axis = 0)
        self.method = method
        self.normalize(dataset)
    def normalize(self, dataset):
        data = dataset.data
        if self.method == "rescaling":
            #Rescaling method
            for i in range(0,self.size[1]-1):
                data[:,i] = (data[:,i] - self.min[i]) / (self.max[i] - self.min[i])
        else:
            # Standarization method
            for i in range(0, self.size[1] - 1):
                data[:, i] = divide((data[:, i] - self.mean[i]), self.std[i])