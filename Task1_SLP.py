import numpy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


class Task1SLPerceptron:
    def __init__(self,
                 Feature1='', Feature2='', Bias_Exist=0,
                 Class1='', loop=50, Class2='', alpha=0.05):
        self.InputX = None
        self.alpha = alpha
        self.InputY = None
        self.itr = loop
        self.TestX = None
        self.bias = Bias_Exist
        self.TrainX = None
        self.Input_weights = np.zeros(2)
        self.weight = [np.random.rand(1)[0], np.random.rand(1)[0], np.random.rand(1)[0]]
        self.bias = np.random.rand()
        self.TestY = None
        self.TrainY = None
        self.FeatureOne = Feature1
        self.ClassOne = Class1
        self.FeatureTwo = Feature2
        self.ClassTwo = Class2

    def Data_Cleaning(self):
        pen = pd.read_csv('penguins.csv')
        ExtractedClasses = ['Adelie', 'Gentoo', 'Chinstrap']
        ExtractedClasses.remove(self.ClassOne)
        ExtractedClasses.remove(self.ClassTwo)
        sex_mapping = {"male": 0, "female": 1}  # map each Sex value to a numerical value
        pen['gender'] = pen['gender'].map(sex_mapping)
        pen["gender"].fillna(pen["gender"].value_counts().idxmax(), inplace=True)
        pen = pen[pen.species != ExtractedClasses[0]]
        pen['species'] = pen['species'].replace(self.ClassOne, -1)
        pen['species'] = pen['species'].replace(self.ClassTwo, 1)
        self.InputX = np.array(pen[[self.FeatureOne, self.FeatureTwo]])
        self.InputY = np.array(pen['species'])
        species = ['Adelie', 'Gentoo', 'Chinstrap']
        Cls1 = 0
        Cls2 = 0
        for idx in species:
            if self.ClassOne == species[0]:
                self.peng = pen.iloc[0:50]
                x1_c1 = self.peng[self.FeatureOne]
                x2_c1 = self.peng[self.FeatureTwo]
                self.c1 = 1
            elif self.ClassOne == species[1]:
                self.peng = pen.iloc[50:100]
                x1_c1 = self.peng[self.FeatureOne]
                x2_c1 = self.peng[self.FeatureTwo]
                self.c1 = 2
            else:
                self.peng = pen.iloc[100:]
                x1_c1 = self.peng[self.FeatureOne]
                x2_c1 = self.peng[self.FeatureTwo]
                self.c1 = 3

        for idx in species:
            if self.ClassTwo == species[0]:
                self.peng2 = pen.iloc[0:50]
                x1_c2 = self.peng2[self.FeatureOne]
                x2_c2 = self.peng2[self.FeatureTwo]
                self.c2 = 1
            elif self.ClassTwo == species[1]:
                self.peng2 = pen.iloc[50:100]
                x1_c2 = self.peng2[self.FeatureOne]
                x2_c2 = self.peng2[self.FeatureTwo]
                self.c2 = 2
            else:
                self.peng2 = pen.iloc[100:]
                x1_c2 = self.peng2[self.FeatureOne]
                x2_c2 = self.peng2[self.FeatureTwo]
                self.c2 = 3
        Cls1 = self.c1
        Cls2 = self.c2
        print("The Classes are chosen : ", Cls1, Cls2)
        print("Feature 1 - Class 1:", x1_c1)
        print("Feature 1 - Class 2:", x1_c2)
        print("Feature 2 - Class 1:", x2_c1)
        print("Feature 2 - Class 2:", x2_c2)

        x_c1 = np.array((x1_c1, x2_c1), dtype=float)
        x_c1 = np.transpose(x_c1)
        x_c2 = np.array((x1_c2, x2_c2), dtype=float)
        x_c2 = np.transpose(x_c2)
        y_c1 = np.full(50, 1)
        y_c2 = np.full(50, -1)

        self.TrainX, self.TestX, self.TrainY, self.TestY = train_test_split(self.InputX, self.InputY, test_size=0.4,
                                                                            shuffle=True,
                                                                            random_state=123)
        print(self.TrainX)
        print(self.TestX)
        print(self.TrainY)
        print(self.TestY)

    def NInput_Calculation(self, X, weight, bias):
        if self.bias == 1:
            v = bias * weight[0] + X[0] * weight[1] + weight[2] * X[1]
        else:
            v = weight[0] + weight[1] * X[0] + weight[2] * X[1]

        return v

    def Learning_Calculation(self):
        Upd_weights = [1, 1, 1]
        NB = self.bias * self.bias
        for i in range(self.itr):
            for j in range(0, len(self.TrainX)):
                w = self.Input_weights
                x = [self.TrainX[j][0], self.TrainX[j][1]]
                Indication = self.TrainY[j]
                v = self.NInput_Calculation(x, weight=Upd_weights, bias=NB)
                # net_value=np.dot(self.x_tran,n)+self.bias
                if v >= 0.0:
                    actual = 1
                else:
                    actual = -1
                error = Indication - actual
                Upd_weights[0] = Upd_weights[0] + self.alpha * error
                Upd_weights[1] = Upd_weights[1] + self.alpha * error * x[0]
                Upd_weights[2] = Upd_weights[2] + self.alpha * error * x[1]
                self.itr = self.itr + 1
                w1 = Upd_weights[1]
                w2 = Upd_weights[2]
                b = Upd_weights[0] * NB
                NB = b
                Upd_weights = [Upd_weights[0], w1, w2]
                self.Input_weights = Upd_weights

        return Upd_weights, NB

    def Testing_Calculations(self, new_weights, NB):
        conf = np.zeros([2, 2], dtype='int32')
        arr_of_predict = []
        for j in range(0, len(self.TestX)):
            x = [self.TestX[j][0], self.TestX[j][1]]
            label = self.TestY[j]
            v = self.NInput_Calculation(x, weight=new_weights, bias=NB)
            if v >= 0.0:
                actual = 1
            else:
                actual = -1

            if actual == 1 and self.TestY[j] == 1:
                conf[0, 0] = conf[0, 0] + 1
            elif actual == -1 and self.TestY[j] == -1:
                conf[1, 1] = conf[1, 1] + 1
            elif actual == 1 and self.TestY[j] == -1:
                conf[0, 1] = conf[0, 1] + 1
            else:
                conf[1, 0] = conf[1, 0] + 1
            arr_of_predict.append(actual)
            tp = conf[0][0]
            tn = conf[1][1]
            fp = conf[0][1]
            fn = conf[1][0]
        Accuracy = (tp + tn) / (tp + tn + fn + fp) * 100
        print("Perceptron Classification accuracy", Accuracy)
        return conf, arr_of_predict

    def DBLine_Plotting(self, y_predict):
        InputFeatureOne = self.FeatureOne
        InputFeatureTwo = self.FeatureTwo
        FXPoint = [np.min(self.TestX[:, 0] - 30), np.max(self.TestX[:, 1] + 30)]
        FYPoint = np.dot((-1 / self.Input_weights[2]), (np.dot(self.Input_weights[1], FXPoint) + self.bias))
        plt.scatter(self.TestX[:, 0], self.TestX[:, 1], marker="o", c=self.TestY)
        plt.plot(FXPoint, FYPoint, label='Separating the data by Decision Boundary')
        plt.title("Separate Two Classes")
        plt.xlabel(InputFeatureOne)
        plt.ylabel(InputFeatureTwo)
        plt.show()

    def CM(self, ActData, PredData):
        d = {}
        key = zip(ActData, PredData)
        for t, a in key:
            if (t, a) in d:
                d[(t, a)] += 1
            else:
                d[(t, a)] = 1
        ind = pd.MultiIndex.from_tuples(d.keys())
        testedValues = pd.Series(index=ind, data=list(d.values()))
        df = testedValues.unstack().fillna(0)
        return df

    def CM_Plotting(self, Actual_L, Predicted_L):
        HM = sns.heatmap(self.CM(Actual_L, Predicted_L), xticklabels=self.CM(Actual_L, Predicted_L).index,
                         yticklabels=self.CM(Actual_L, Predicted_L).columns, annot=True, )
        HM.invert_yaxis()
        plt.ylabel("Target Values")
        plt.xlabel("Predicted Values")
        plt.show()
