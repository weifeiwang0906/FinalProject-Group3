'''
In the application we will implement controls and triggers
the controls that oversee here are:
   Checkbox
   TextBox
   Radio bottoms
'''

import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication

from PyQt5.QtWidgets import QCheckBox    # checkbox
from PyQt5.QtWidgets import QPushButton  # pushbutton
from PyQt5.QtWidgets import QLineEdit    # Lineedit
from PyQt5.QtWidgets import QRadioButton # Radio Buttons

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt  # Control status
from PyQt5.QtWidgets import  QWidget,QLabel, QVBoxLayout, QHBoxLayout, QGridLayout

#::------------------------------------------------------------------------------------
#:: Class: Check Control
#::------------------------------------------------------------------------------------
class CheckControlClass(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(CheckControlClass, self).__init__()

        self.Title = 'Title : Final texts show '
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout )
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout

        Cbox = QCheckBox('Show the clean text', self)
        Cbox.move(20, 20)
        Cbox.stateChanged.connect(self.showchange)

        Cbox2 = QCheckBox('Show the clean text2', self)
        Cbox2.move(20, 20)
        Cbox2.stateChanged.connect(self.showchange2)

        self.label11 = QLabel(
            "Original text: #AFRICANBAZE: Breeeeaking news:Nigeria flag set ablaze in Aba People didn't know the reason. http://t.co/2nndBGwyEi")
        self.label22 = QLabel(
            "Original text: During the 1960s the oryx a symbol of the Arabian Peninsula were annihilated by hunters. http://t.co/yangEQBUQW http://t.co/jQ2eH5KGLt")

        self.setGeometry(300, 300, 250, 150)

        self.layout.addWidget(Cbox)
        self.layout.addWidget(self.label11)
        self.layout.addWidget(Cbox2)
        self.layout.addWidget(self.label22)
        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(1000, 200)                         # Resize the window


    def showchange(self, state):
        if state == Qt.Checked:
            self.label11.setText("africanbaze breaking news nigeria flag set ablaze aba people know reason")
        else:
            self.label11.setText("#AFRICANBAZE: Breeeeaking news:Nigeria flag set ablaze in Aba People didn't know the reason. http://t.co/2nndBGwyEi")

    def showchange2(self, state):
        if state == Qt.Checked:
            self.label22.setText("1960s oryx symbol arabian peninsula annihilated hunters")
        else:
            self.label22.setText("During the 1960s the oryx a symbol of the Arabian Peninsula were annihilated by hunters. http://t.co/yangEQBUQW http://t.co/jQ2eH5KGLt")

#::------------------------------------------------------------------------------------
#:: Class: Radio Button
#::------------------------------------------------------------------------------------
class RadioButtonClass(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(RadioButtonClass, self).__init__()

        self.Title = 'Title : Preprocess Instance Show '
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout )
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)   # Creates vertical layout

        self.label1 = QLabel("Original text: #AFRICANBAZE: Breeeeaking news:Nigeria flag set ablaze in Aba People didn't know the reason. http://t.co/2nndBGwyEi")
        self.layout.addWidget(self.label1,0, 0, 1, 3)

        self.b6 = QRadioButton("Mislabel")
        self.b6.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b6,2,0)

        self.b1 = QRadioButton("Contraction map")
        self.b1.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b1,2,1)

        self.b2 = QRadioButton("http://")
        self.b2.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b2,2,2)

        self.b7 = QRadioButton("Repetitive Letter")
        self.b7.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b7,2,3)

        self.b8 = QRadioButton("abbreviation")
        self.b8.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b8,3,0)

        self.b3 = QRadioButton("Lowercase")
        self.b3.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b3,3,1)

        self.b4 = QRadioButton("Punctuation")
        self.b4.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b4,3,2)

        self.b5 = QRadioButton("Stopwords")
        self.b5.toggled.connect(self.onClicked)
        self.layout.addWidget(self.b5,3,3)

        self.buttonlabel= QLabel('   <Converting result shows here>',self)
        self.layout.addWidget(self.buttonlabel,4,0,1,4)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(800, 150)                         # Resize the window


    def onClicked(self):
        button = self.sender()
        if button.isChecked():
            if button.text()=='Mislabel':
                self.buttonlabel.setText("   Result:    Instance isn't included by this part,this part aims for manually relabeling the wrong targets")
            elif button.text() == 'Contraction map':
                self.buttonlabel.setText("   Result:    #AFRICANBAZE: Breeeeaking news:Nigeria flag set ablaze in Aba People did not know the reason. http://t.co/2nndBGwyEi")
            elif button.text() == 'http://':
                self.buttonlabel.setText("   Result:    #AFRICANBAZE: Breeeeaking news:Nigeria flag set ablaze in Aba People did not know the reason.")
            elif button.text() == 'Repetitive Letter':
                self.buttonlabel.setText("   Result:    #AFRICANBAZE: Breaking news:Nigeria flag set ablaze in Aba People didn't know the reason. http://t.co/2nndBGwyEi")
            elif button.text() == 'abbreviation':
                self.buttonlabel.setText(
                    "   Result:    Instance isn't included by this part,this part is same as contraction map, but they use different dictionaries")
            elif button.text() == 'Lowercase':
                self.buttonlabel.setText("   Result:    #africanbaze: breeeeaking news:nigeria flag set ablaze in aba people didn't know the reason. http://t.co/2nndbgwyei")
            elif button.text() == 'Punctuation':
                self.buttonlabel.setText("   Result:    AFRICANBAZE Breeeeaking news Nigeria flag set ablaze in Aba People didnt know the reason httptco2nndBGwyEi")
            elif button.text()=='Stopwords':
                self.buttonlabel.setText("   Result:    #AFRICANBAZE: Breeeeaking news:Nigeria flag set ablaze Aba People  know reason. http://t.co/2nndBGwyEi")






#::-------------------------------------------------------------
#:: Definition of a Class for the main manu in the application
#::-------------------------------------------------------------
class Menu(QMainWindow):

    def __init__(self):

        super().__init__()
        #::-----------------------
        #:: variables use to set the size of the window that contains the menu
        #::-----------------------
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 300

        #:: Title for the application

        self.Title = '6103 project Preprocessing'

        #:: The initUi is call to create all the necessary elements for the menu

        self.initUI()

    def initUI(self):

        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()
        #::-----------------------------
        # 1. Create the menu bar
        # 2. Create an item in the menu bar
        # 3. Creaate an action to be executed the option in the  menu bar is choosen
        #::-----------------------------
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')

        #:: Add another option to the Menu Bar

        exampleWin = mainMenu.addMenu ('preprocess')

        #::--------------------------------------
        # Exit action
        # The following code creates the the da Exit Action along
        # with all the characteristics associated with the action
        # The Icon, a shortcut , the status tip that would appear in the window
        # and the action
        #  triggered.connect will indicate what is to be done when the item in
        # the menu is selected
        # These definitions are not available until the button is assigned
        # to the menu
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        #:: This line adds the button (item element ) to the menu

        fileMenu.addAction(exitButton)

        #::------------------------------------------------------------
        #:: Add code to include radio buttons  to implement an action upon request
        #::------------------------------------------------------------

        exampleRadioButton = QAction("Instance", self)
        exampleRadioButton.setStatusTip('Example of preproceing')
        exampleRadioButton.triggered.connect(self.ExampleRadioButton)

        exampleWin.addAction(exampleRadioButton)

        examplefinalbutton=QAction('Final Compare',self)
        examplefinalbutton.setStatusTip('To show the original text and clean text.')
        examplefinalbutton.triggered.connect(self.Examplefinalbutton)

        exampleWin.addAction(examplefinalbutton)
        #:: Creates an empty list of dialogs to keep track of
        #:: all the iterations

        self.dialogs = list()

        #:: This line shows the windows
        self.show()

    def ExampleRadioButton(self):
        dialog = RadioButtonClass()
        self.dialogs.append(dialog) # Apppends to the list of dialogs
        dialog.show()

    def Examplefinalbutton(self):
        dialog = CheckControlClass()
        self.dialogs.append(dialog) # Apppends to the list of dialogs
        dialog.show()
#::------------------------
#:: Application starts here
#::------------------------

def main():
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = Menu()  # Cretes the menu
    sys.exit(app.exec_())  # Close the application

if __name__ == '__main__':
    main()