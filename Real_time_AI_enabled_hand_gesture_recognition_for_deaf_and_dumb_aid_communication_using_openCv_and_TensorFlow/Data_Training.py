import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np

class RandomForestModelTrainer:
    def __init__(self, data_filename):
        self.data_filename = data_filename
        self.model = RandomForestClassifier()

    def load_data(self):
        data_dict = pickle.load(open(self.data_filename, 'rb'))
        self.data = np.asarray(data_dict['data'])
        self.labels = np.asarray(data_dict['labels'])

    def train_model(self, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=test_size, shuffle=True, stratify=self.labels)
        self.model.fit(x_train, y_train)
        self.x_test = x_test
        self.y_test = y_test

    def evaluate_model(self):
        y_predict = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_predict)
        precision = precision_score(self.y_test, y_predict, average='weighted')
        f1 = f1_score(self.y_test, y_predict, average='weighted')
        return accuracy, precision, f1

    def save_model(self, model_filename):
        with open(model_filename, 'wb') as f:
            pickle.dump({'model': self.model}, f)

if __name__ == "__main__":
    data_filename = './data.pickle'
    model_trainer = RandomForestModelTrainer(data_filename)

    model_trainer.load_data()
    model_trainer.train_model()
    accuracy, precision, f1 = model_trainer.evaluate_model()

    model_filename = 'model.p'
    model_trainer.save_model(model_filename)

    print('Accuracy: {:.2%}'.format(accuracy))
    print('Precision (weighted average): {:.2%}'.format(precision))
    print('F1-Score (weighted average): {:.2%}'.format(f1))

