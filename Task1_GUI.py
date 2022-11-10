import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Task1_SLP as s
from tkinter import messagebox
import tkinter.ttk as ttk
import tkinter
from tkinter import *
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class gui:
    def __init__(self):
        global Indications
        df = pd.read_csv("penguins.csv")
        self.DATA = df
        Indications = ['Adelie', 'Gentoo', 'Chinstrap']
        global features
        features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']
        self.Window = Tk()
        self.Window.title('SLP Task 1')
        self.Window.geometry("550x350")
        self.Window.configure(bg="#FFFFFF")
        self.canvas = Canvas(self.Window, bg="#669999", height=600, width=900, bd=0, highlightthickness=0,
                             relief="ridge")
        self.canvas.place(x=0, y=0)

        # class 1---------------
        self.class1 = ttk.Combobox(self.Window, state="readonly", values=Indications)
        self.class1.place(x=85, y=56, width=85, height=30)
        Label(self.Window, text="Class 1", bg='white').place(x=40, y=60)

        # class2--------------

        self.class2 = ttk.Combobox(self.Window, state="readonly", values=Indications)
        self.class2.place(x=280, y=56, width=85, height=30)
        self.class2.bind()
        Label(self.Window, text="Class 2", bg='white').place(x=220, y=60)

        # featyre 1----------
        self.feature1 = ttk.Combobox(self.Window, state="readonly", values=features)
        self.feature1.place(x=85, y=110, width=130, height=30)
        Label(self.Window, text="Feature 1", bg='white').place(x=26, y=110)

        # feature2------
        self.feature2 = ttk.Combobox(self.Window, state="readonly", values=features)
        self.feature2.place(x=300, y=110, width=130, height=30)
        Label(self.Window, text="Feature 2", bg='white').place(x=240, y=110)

        # learning rate
        alpha = tkinter.StringVar()
        alpha.set("")
        loop = tkinter.StringVar()
        loop.set("")
        Label(self.Window, text="Learning_Rate", bg='white').place(x=15, y=170)

        self.alpha_entry = Entry(bd=0, bg="#d9d9d9", highlightthickness=0, textvariable=alpha)
        self.alpha_entry.place(x=100, y=170, width=40, height=28)

        # epochs -------
        self.epochs_entry = Entry(bd=0, bg="#d9d9d9", highlightthickness=0, textvariable=loop)
        self.epochs_entry.place(x=330, y=170, width=40, height=28)
        Label(self.Window, text="Number_of_Epochs", bg='white').place(x=210, y=170)

        global bias
        bias = IntVar()
        self.bias_entry = ttk.Checkbutton(self.Window, variable=bias, onvalue=1, offvalue=0)
        self.bias_entry.configure(text='Bias')
        self.bias_entry.place(x=50, y=210, width=65, height=28)

        self.click_button = Button(text="Run", command=self.Run)
        self.click_button.place(x=230, y=240, width=65, height=40)

        self.click_button = Button(text="Plotting", command=self.Pretrained_Plotting)
        self.click_button.place(x=120, y=240, width=65, height=40)

        self.close_button = Button(text="Close", command=self.close)
        self.close_button.place(x=330, y=240, width=65, height=40)
        self.Window.mainloop()

    def Run(self):
        Feature1 = self.feature1.get()
        eta = self.alpha_entry.get()
        Feature2 = self.feature2.get()
        Bias_Exist = bias.get()
        loop = self.epochs_entry.get()
        c1 = self.class1.get()
        c2 = self.class2.get()
        loop = int(loop)
        eta = float(eta)

        conf = np.zeros([2, 2], dtype='int32')
        perceptron = s.Task1SLPerceptron(alpha=eta, loop=loop, Bias_Exist=Bias_Exist, Class1=c1,
                                         Class2=c2,
                                         Feature1=Feature1, Feature2=Feature2)
        perceptron.Data_Cleaning()
        w, b = perceptron.Learning_Calculation()
        conf, y_predict = perceptron.Testing_Calculations(w, b)
        print(conf)
        perceptron.CM_Plotting(perceptron.TestY, y_predict)
        perceptron.DBLine_Plotting(y_predict)

    def plot_cl0(self):
        pen = self.DATA
        sns.scatterplot(x="bill_length_mm", y="flipper_length_mm", data=pen, hue="species")
        # plt.title("bill_length_mm  vs flipper_length_mm", size=20, color="red")
        plt.show()

    def plot_cl1(self):
        pen = self.DATA
        encode = LabelEncoder()
        pen['gender'] = encode.fit_transform(pen['gender'])
        pen['gender'] = pen['gender'].replace(2, pen['gender'].idxmax())
        sns.scatterplot(x="bill_length_mm", y="gender", data=pen, hue="species")
        plt.show()

    def plot_cl2(self):
        pen = self.DATA
        sns.scatterplot(x="bill_length_mm", y="bill_depth_mm", data=pen, hue="species")
        # plt.title("bill_length_mm  vs bill_depth_mm", size=20, color="red")
        plt.show()

    def plot_cl3(self):
        pen = self.DATA
        sns.scatterplot(x="bill_length_mm", y="body_mass_g", data=pen, hue="species")
        # plt.title("bill_length_mm  vs body_mass_g", size=20, color="red")
        plt.show()

    def plot_cl4(self):
        pen = self.DATA
        sns.scatterplot(x="flipper_length_mm", y="bill_depth_mm", data=pen, hue="species")
        # plt.title("flipper_length_mm  vs bill_depth_mm", size=20, color="red")
        plt.show()

    def plot_cl5(self):
        pen = self.DATA
        sns.scatterplot(x="flipper_length_mm", y="body_mass_g", data=pen, hue="species")
        # plt.title("flipper_length_mm  vs body_mass_g", size=20, color="red")
        plt.show()

    def plot_cl6(self):
        pen = self.DATA
        encode = LabelEncoder()
        pen['gender'] = encode.fit_transform(pen['gender'])
        pen['gender'] = pen['gender'].replace(2, pen['gender'].idxmax())
        sns.scatterplot(x="flipper_length_mm", y="gender", data=pen, hue="species")
        # plt.title("flipper_length_mm  vs gender", size=20, color="red")
        plt.show()

    def plot_cl7(self):
        pen = self.DATA
        sns.scatterplot(x="bill_depth_mm", y="body_mass_g", data=pen, hue="species")
        # plt.title("bill_depth_mm  vs body_mass_g", size=20, color="red")
        plt.show()

    def plot_cl8(self):
        pen = self.DATA
        encode = LabelEncoder()
        pen['gender'] = encode.fit_transform(pen['gender'])
        pen['gender'] = pen['gender'].replace(2, pen['gender'].idxmax())
        sns.scatterplot(x="bill_depth_mm", y="gender", data=pen, hue="species")
        # plt.title("bill_depth_mm  vs gender", size=20, color="red")
        plt.show()

    def plot_cl9(self):
        pen = self.DATA
        encode = LabelEncoder()
        pen['gender'] = encode.fit_transform(pen['gender'])
        pen['gender'] = pen['gender'].replace(2, pen['gender'].idxmax())
        sns.scatterplot(x="body_mass_g", y="gender", data=pen, hue="species")
        plt.show()

    def close(self):
        self.Window.destroy()

    def Pretrained_Plotting(self):
        self.plot_cl0()
        self.plot_cl1()
        self.plot_cl2()
        self.plot_cl3()
        self.plot_cl4()
        self.plot_cl5()
        self.plot_cl6()
        self.plot_cl7()
        self.plot_cl8()
        self.plot_cl9()


model = gui()
