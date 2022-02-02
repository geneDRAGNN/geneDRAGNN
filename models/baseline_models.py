from numpy import mean, std
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import json

class KNNBaselineModel:
    def __init__(self, graph, features):
        self.graph = graph
        self.features = features
        self.data = pd.DataFrame()
        self.reset()
        self.extract_dataset()

    def reset(self):
        self.accuracies = []
        self.results = []
        self.best_model = KNeighborsClassifier()
        self.best_acc = 0
    
    def extract_dataset(self):
        if self.features not in ['all', 'node', 'network']:
            raise ValueError("Unexpected feature set. Try feat=\'all\', \'node\', \'network\'")

        df = pd.read_csv(f"../Datasets/{self.graph}_graph/final_all_node_data_weighted_{self.graph}_v1.csv")

        df = df.drop(list(df.filter(regex = 'gda')), axis = 1)

        if self.features == "node":
            df = df.drop(list(df.filter(regex = 'network')), axis = 1)
        elif self.features == "network":
            for x in ['hpa', 'nih', 'nx']:
                df = df.drop(list(df.filter(regex = x)), axis = 1)

        return df

    def model_test(self, x, y):
        x_train_val, x_test, y_train_val, y_test = train_test_split(x.values, y.values, test_size=0.2, stratify=y.values)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.125, stratify=y_train_val)

        classifier = KNeighborsClassifier()
        classifier.fit(x_train, y_train)

        y_pred_val = classifier.predict(x_test)
        test_report = classification_report(y_test, y_pred_val, output_dict=True) 
        accuracy = accuracy_score(y_test, y_pred_val)

        if accuracy > self.best_acc:
            self.best_model = classifier
            self.best_acc = accuracy

        self.results.append(test_report)
        self.accuracies.append(accuracy)

        return test_report, accuracy

    def run_trials(self, thres, start=0, end=100):
        self.reset()
        self.thres = thres

        labels = pd.read_csv(f"../Datasets/{self.graph}_graph/labels_thres{thres}_trials_{self.graph}.csv")
        labels = labels.drop(list(labels.filter(regex = 'gda')), axis = 1)

        print(f"\nRunning {end-start} trials with the {self.graph} graph, {thres} threshold, and {self.features} features.", end='')
        for i in tqdm(range(start, end)):
            trial = self.data.merge(labels[['ensembl', f'label_{i}']], on='ensembl').dropna()
            self.model_test(trial.iloc[:, 1:-1], trial.iloc[:, -1])

    def print_results(self, to_json=False, filename=None, to_txt=False):
        print(f"KNN - {self.graph} - {self.thres} - {self.features}")
        print("\tTrials:", len(self.accuracies))
        print("\tAvg. Accuracy:", mean(self.accuracies))
        print("\tStandard Deviation:", std(self.accuracies))

        if to_json == True and filename != None:
            with open(f'results/{filename}.json', "w") as file:
                json.dump(self.results, file)
        elif to_json == True:
            if self.weighted == False:
                with open(f'results/knn_{self.graph}graph_{self.features}_{self.thres}.json', "w") as file:
                    json.dump(self.results, file)
        
        if to_txt == True:
            with open("resultsKNN.txt", "a") as myfile:
                myfile.write(f"KNN - {self.features}")
                myfile.write(f"\n\tTrials: {len(self.accuracies)}")
                myfile.write(f"\n\tAvg. Accuracy: {mean(self.accuracies)}")
                myfile.write(f"\n\tStandard Deviation: {std(self.accuracies)}\n\n")