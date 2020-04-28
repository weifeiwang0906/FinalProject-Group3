##################################################
### Created by Yu Cao
### Project Name : Real or Not? NLP with Disaster Tweets
### Date 04/27/2020
### Data Mining
##################################################

import sys

#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

from scipy import interp
from itertools import cycle


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import pydot
from numpy.polynomial.polynomial import polyfit

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import random
import seaborn as sns

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
#%%-----------------------------------------------------------------------


#::--------------------------------
# Deafault font size for all the windows
#::--------------------------------
font_size_window = 'font-size:15px'

class KaggleRF(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the preprocessed dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KaggleRF, self).__init__()
        self.Title = "Kaggle Score of Random Forest"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid layout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.scoretxt_rf = QLabel('Kaggle Score of Random Forest:')
        self.scoretxt_rf.move(50, 20)


        self.png_rf = QPixmap('/Users/apple/Documents/me/GWU/courses/ds6103/python03/Data-Mining-master/final project/gui/test_kaggle_rf.png')
        self.scoretxt_rf.setPixmap(self.png_rf)
        self.scoretxt_rf.setScaledContents(True)


        self.layout.addWidget(self.scoretxt_rf)
        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)
        self.resize(1500, 50)
        self.show()


class ResultsRF(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the preprocessed dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(ResultsRF, self).__init__()
        self.Title = "Results of Random Forest"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid layout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        # predicton on test using all features
        y_pred = clf_rf.predict(X_test)
        y_pred_score = clf_rf.predict_proba(X_test)

        # clasification report
        self.rf_class_rep = classification_report(y_test, y_pred)
#        self.txtResults.appendPlainText(self.rf_class_rep)

        # accuracy score
        self.rf_accuracy_score = accuracy_score(y_test, y_pred) * 100
#        self.txtAccuracy.setText(str(self.rf_accuracy_score))

        self.lblResults = QLabel('Classification:')
        self.txtResults = QPlainTextEdit()
        self.txtResults.setPlainText(self.rf_class_rep)
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()
        self.txtAccuracy.setText(np.str(self.rf_accuracy_score))

        self.layout.addWidget(self.lblResults)
        self.layout.addWidget(self.txtResults)
        self.layout.addWidget(self.lblAccuracy)
        self.layout.addWidget(self.txtAccuracy)
        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)
        self.resize(400, 300)

class ROCcurveRF(QMainWindow):
#::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the preprocessed dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(ROCcurveRF, self).__init__()
        self.Title = "ROC Curve of Random Forest"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.rf_probs = clf_rf.predict_proba(X_test)
        self.rf_probs = self.rf_probs[:, 1]

        self.rf_auc = roc_auc_score(y_test, self.rf_probs)

        self.rf_fpr, self.rf_tpr, _ = roc_curve(y_test, self.rf_probs)

        self.ax1.plot(self.rf_fpr, self.rf_tpr, color='darkorange',
                      lw=2, label='ROC curve (area = %0.3f)' % self.rf_auc)
        self.ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        self.ax1.set_xlim([0.0, 1.0])
        self.ax1.set_ylim([0.0, 1.05])
        self.ax1.set_xlabel('False Positive Rate')
        self.ax1.set_ylabel('True Positive Rate')
        self.ax1.set_title('ROC Curve Random Forest')
        self.ax1.legend(loc="lower right")

        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        self.layout.addWidget(self.canvas1)

        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)
        self.resize(400, 350)
        self.show()

class PRcurveRF(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the preprocessed dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(PRcurveRF, self).__init__()
        self.Title = "P-R Curve of Random Forest"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        y_pred_score = clf_rf.predict_proba(X_test)
        y_pred_score = y_pred_score[:, 1]

        self.rf_precision, self.rf_recall, _ = precision_recall_curve(y_test, y_pred_score)
        self.rf_auc_pr = auc(self.rf_recall, self.rf_precision)
        self.ns_pr_rf = len(y_test[y_test == 1]) / len(y_test)
        self.ax2.plot(self.rf_recall, self.rf_precision, color='darkorange',
                      lw=2, label='P-R curve (area = %0.3f)' % self.rf_auc_pr)
        self.ax2.plot([0, 1], [self.ns_pr_rf, self.ns_pr_rf], color='navy', lw=2, linestyle='--')

        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_xlabel('Recall')
        self.ax2.set_ylabel('Precision')
        self.ax2.set_title('P-R Curve Random Forest')
        self.ax2.legend(loc="lower right")

        # show the plot
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        self.layout.addWidget(self.canvas2)

        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)
        self.resize(400, 350)
        self.show()


class KaggleNB(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the preprocessed dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KaggleNB, self).__init__()
        self.Title = "Kaggle Score of Naive Bayes"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid layout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.scoretxt_nb = QLabel('Kaggle Score of Naive Bayes:')
        self.scoretxt_nb.move(50, 20)


        self.png_nb = QPixmap('/Users/apple/Documents/me/GWU/courses/ds6103/python03/Data-Mining-master/final project/gui/test_kaggle_nb.png')
        self.scoretxt_nb.setPixmap(self.png_nb)


        self.layout.addWidget(self.scoretxt_nb)
        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)
        self.resize(600, 200)
        self.show()



class ResultsNB(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the preprocessed dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(ResultsNB, self).__init__()
        self.Title = "Results of Naive Bayes"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid layout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        # prediction on test using all features
        y_pred_nb = clf_nb.predict(X_test)
        y_pred_score_nb = clf_nb.predict_proba(X_test)

        # classification report
        self.nb_class_rep = classification_report(y_test, y_pred_nb)
        #        self.txtResults.appendPlainText(self.rf_class_rep)

        # accuracy score
        self.nb_accuracy_score = accuracy_score(y_test, y_pred_nb) * 100
        #        self.txtAccuracy.setText(str(self.rf_accuracy_score))

        self.lblResults_nb = QLabel('Classification:')
        self.txtResults_nb = QPlainTextEdit()
        self.txtResults_nb.setPlainText(self.nb_class_rep)
        self.lblAccuracy_nb = QLabel('Accuracy:')
        self.txtAccuracy_nb = QLineEdit()
        self.txtAccuracy_nb.setText(np.str(self.nb_accuracy_score))

        self.layout.addWidget(self.lblResults_nb)
        self.layout.addWidget(self.txtResults_nb)
        self.layout.addWidget(self.lblAccuracy_nb)
        self.layout.addWidget(self.txtAccuracy_nb)
        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)
        self.resize(400, 300)


class ROCcurveNB(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the preprocessed dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(ROCcurveNB, self).__init__()
        self.Title = "ROC Curve of Naive Bayes"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.nb_probs = clf_nb.predict_proba(X_test)
        self.nb_probs = self.nb_probs[:, 1]

        self.nb_auc = roc_auc_score(y_test, self.nb_probs)

        self.nb_fpr, self.nb_tpr, _ = roc_curve(y_test, self.nb_probs)

        self.ax3.plot(self.nb_fpr, self.nb_tpr, color='darkorange',
                      lw=2, label='ROC curve (area = %0.3f)' % self.nb_auc)
        self.ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        self.ax3.set_xlim([0.0, 1.0])
        self.ax3.set_ylim([0.0, 1.05])
        self.ax3.set_xlabel('False Positive Rate')
        self.ax3.set_ylabel('True Positive Rate')
        self.ax3.set_title('ROC Curve Naive Bayes')
        self.ax3.legend(loc="lower right")

        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        self.layout.addWidget(self.canvas3)

        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)
        self.resize(400, 350)
        self.show()


class PRcurveNB(QMainWindow):
    #::--------------------------------------------------------------------------------
    # Implementation of Random Forest Classifier using the preprocessed dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas
    #::---------------------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        super(PRcurveNB, self).__init__()
        self.Title = "P-R Curve of Naive Bayes"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.fig4 = Figure()
        self.ax4 = self.fig4.add_subplot(111)
        self.canvas4 = FigureCanvas(self.fig4)

        self.canvas4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas4.updateGeometry()

        y_pred_score = clf_nb.predict_proba(X_test)
        y_pred_score = y_pred_score[:, 1]

        self.nb_precision, self.nb_recall, _ = precision_recall_curve(y_test, y_pred_score)
        self.nb_auc_pr = auc(self.nb_recall, self.nb_precision)
        self.ns_pr_nb = len(y_test[y_test == 1]) / len(y_test)
        self.ax4.plot(self.nb_recall, self.nb_precision, color='darkorange',
                      lw=2, label='P-R curve (area = %0.3f)' % self.nb_auc_pr)
        self.ax4.plot([0, 1], [self.ns_pr_nb, self.ns_pr_nb], color='navy', lw=2, linestyle='--')

        self.ax4.set_xlim([0.0, 1.0])
        self.ax4.set_xlabel('Recall')
        self.ax4.set_ylabel('Precision')
        self.ax4.set_title('P-R Curve Naive Bayes')
        self.ax4.legend(loc="lower right")

        # show the plot
        self.fig4.tight_layout()
        self.fig4.canvas.draw_idle()

        self.layout.addWidget(self.canvas4)

        self.setGeometry(300, 300, 250, 150)

        self.setCentralWidget(self.main_widget)
        self.resize(400, 350)
        self.show()


class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 300
        self.Title = 'Real or not? NLP with Disaster Tweets'
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        RFMenu = mainMenu.addMenu('Random Forest')
        NBMenu = mainMenu.addMenu('Naive Bayes')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are two models
        #       Random Forest
        #       Naive Bayes
        #::------------------------------------------------------
        # Random Forest Classifier -- Results
        #::------------------------------------------------------
        KaggleRFButton = QAction(QIcon(), 'Kaggle Score of Random Forest', self)
        KaggleRFButton.setStatusTip('Kaggle Score of Random Forest')
        KaggleRFButton.triggered.connect(self.KSRF)

        RFMenu.addAction(KaggleRFButton)
        #::------------------------------------------------------
        # Random Forest Classifier -- Results
        #::------------------------------------------------------
        ResultRFButton = QAction(QIcon(), 'Results of Random Forest', self)
        ResultRFButton.setStatusTip('Results of Random Forest')
        ResultRFButton.triggered.connect(self.RERF)

        RFMenu.addAction(ResultRFButton)
        #::------------------------------------------------------
        # Random Forest Classifier -- ROC Curve
        #::------------------------------------------------------
        ROCRFButton = QAction(QIcon(), 'ROC Curve of Random Forest', self)
        ROCRFButton.setStatusTip('ROC Curve of Random Forest')
        ROCRFButton.triggered.connect(self.ROCRF)

        RFMenu.addAction(ROCRFButton)
        #::------------------------------------------------------
        # Random Forest Classifier -- P-R Curve
        #::------------------------------------------------------
        PRRFButton = QAction(QIcon(), 'P-R Curve of Random Forest', self)
        PRRFButton.setStatusTip('P-R Curve of Random Forest')
        PRRFButton.triggered.connect(self.PRRF)

        RFMenu.addAction(PRRFButton)

        #::--------------------------------------------------
        # Random Forest Classifier -- Results
        #::------------------------------------------------------
        KaggleNBButton = QAction(QIcon(), 'Kaggle Score of Naive Bayes', self)
        KaggleNBButton.setStatusTip('Kaggle Score of Naive Bayes')
        KaggleNBButton.triggered.connect(self.KSNB)

        NBMenu.addAction(KaggleNBButton)
        #::--------------------------------------------------
        # Naive Bayes -- Results
        #::--------------------------------------------------
        ResultNBButton =  QAction(QIcon(), 'Results of Naive Bayes', self)
        ResultNBButton.setStatusTip('Results of Naive Bayes')
        ResultNBButton.triggered.connect(self.RENB)

        NBMenu.addAction(ResultNBButton)
        #::------------------------------------------------------
        # Naive Bayes -- ROC Curve
        #::------------------------------------------------------
        ROCNBButton = QAction(QIcon(), 'ROC Curve of Naive Bayes', self)
        ROCNBButton.setStatusTip('ROC Curve of Naive Bayes')
        ROCNBButton.triggered.connect(self.ROCNB)

        NBMenu.addAction(ROCNBButton)
        #::------------------------------------------------------
        # Naive Bayes -- P-R Curve
        #::------------------------------------------------------
        PRNBButton = QAction(QIcon(), 'P-R Curve of Naive Bayes', self)
        PRNBButton.setStatusTip('P-R Curve of Naive Bayes')
        PRNBButton.triggered.connect(self.PRNB)

        NBMenu.addAction(PRNBButton)


        self.dialogs = list()

        self.show()

    def KSRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Results of Random Forest
        #::-------------------------------------------------------------
        dialog = KaggleRF()
        self.dialogs.append(dialog)
        dialog.show()

    def RERF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Results of Random Forest
        #::-------------------------------------------------------------
        dialog = ResultsRF()
        self.dialogs.append(dialog)
        dialog.show()

    def ROCRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the ROC Curve of Random Forest
        #::-------------------------------------------------------------
        dialog = ROCcurveRF()
        self.dialogs.append(dialog)
        dialog.show()

    def PRRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the P-R Curve of Random Forest
        #::-------------------------------------------------------------
        dialog = PRcurveRF()
        self.dialogs.append(dialog)
        dialog.show()

    def KSNB(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Results of Random Forest
        #::-------------------------------------------------------------
        dialog = KaggleNB()
        self.dialogs.append(dialog)
        dialog.show()

    def RENB(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Results of Naive Bayes
        #::-------------------------------------------------------------
        dialog = ResultsNB()
        self.dialogs.append(dialog)
        dialog.show()

    def ROCNB(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the ROC Curve of Naive Bayes
        #::-------------------------------------------------------------
        dialog = ROCcurveNB()
        self.dialogs.append(dialog)
        dialog.show()

    def PRNB(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the P-R Curve of Naive Bayes
        #::-------------------------------------------------------------
        dialog = PRcurveNB()
        self.dialogs.append(dialog)
        dialog.show()


def real_or_not():
    #::--------------------------------------------------
    # Loads the dataset 2017.csv ( Index of happiness and esplanatory variables original dataset)
    # Loads the dataset final_happiness_dataset (index of happiness
    # and explanatory variables which are already preprocessed)
    # Populates X,y that are used in the classes above
    #::--------------------------------------------------
    global ron_train
    global ron_test
    global labels_train
    global features_list_train
    global features_train
    global X_train
    global X_test
    global y_train
    global y_test
    global clf_rf
    global clf_nb

    # read data and preprocess dara
    ron_train = pd.read_csv('/Users/apple/Documents/me/GWU/courses/ds6103/python03/Data-Mining-master/final project/realornot_trainencode.csv')
    ron_test = pd.read_csv('/Users/apple/Documents/me/GWU/courses/ds6103/python03/Data-Mining-master/final project/realornot_testencode.csv')
    ron_train.drop('Unnamed: 0', axis=1, inplace=True)
    ron_test.drop('Unnamed: 0', axis=1, inplace=True)
    labels_train = ron_train.iloc[:, -1]
    features_train = ron_train.iloc[:, 0:-1]
    features_list_train = list(features_train.columns)
    features_train = np.array(features_train)

    # Assign the X and y to run the Random Forest Classifier
    X_dt = ron_train.iloc[:, 0:-1]
    y_dt = ron_train.iloc[:, -1]

    class_le = LabelEncoder()

    # fit and transform the class
    y_dt = class_le.fit_transform(y_dt)

    # split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=100)

    # specify random forest classifier
    clf_rf = RandomForestClassifier(n_estimators=1660, random_state=100)

    # perform training
    clf_rf.fit(X_train, y_train)

    # specify naive bayes classifier
    clf_nb = GaussianNB()
    clf_nb.fit(X_train, y_train)


def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    real_or_not()
    main()
