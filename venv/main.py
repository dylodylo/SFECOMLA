import sys
from PyQt5.uic import loadUiType
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import random
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import wykresy
from matplotlib import pyplot as plt
import numpy as np
import csv
from collections import Counter
import pandas as pd
import datasets
from functools import partial

#import pliku stworzonego w QtDesigner
Ui_MainWindow, QMainWindow = loadUiType('prototyp.ui')


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.addToolBar(NavigationToolbar(self.canvas, self)) #toolbar matlopotlib
        self.pushButton.clicked.connect(self.button_action) #wywołanie buttona "test"
        self.textBrowser.setSource(QUrl("../descriptions/welcome.txt")) #załadowanie tekstu o textBrowser
        self.plot() #rysowanie wykresu

        #wywoływanie informacji o danych/algorytmach/miarach
        self.data_info_1.clicked.connect(partial(self.change_text_source, "../descriptions/data1.txt"))
        self.data_info_2.clicked.connect(partial(self.change_text_source, "../descriptions/data2.txt"))
        self.data_info_3.clicked.connect(partial(self.change_text_source, "../descriptions/data3.txt"))
        self.data_info_4.clicked.connect(partial(self.change_text_source, "../descriptions/data4.txt"))
        self.data_info_5.clicked.connect(partial(self.change_text_source, "../descriptions/data5.txt"))

        self.algorithm_info_1.clicked.connect(partial(self.change_text_source, "../descriptions/algorithm1.txt"))
        self.algorithm_info_2.clicked.connect(partial(self.change_text_source, "../descriptions/algorithm2.txt"))
        self.algorithm_info_3.clicked.connect(partial(self.change_text_source, "../descriptions/algorithm3.txt"))
        self.algorithm_info_4.clicked.connect(partial(self.change_text_source, "../descriptions/algorithm4.txt"))
        self.algorithm_info_5.clicked.connect(partial(self.change_text_source, "../descriptions/algorithm5.txt"))

        self.measure_info_1.clicked.connect(partial(self.change_text_source, "../descriptions/measure1.txt"))
        self.measure_info_2.clicked.connect(partial(self.change_text_source, "../descriptions/measure2.txt"))
        self.measure_info_3.clicked.connect(partial(self.change_text_source, "../descriptions/measure3.txt"))
        self.measure_info_4.clicked.connect(partial(self.change_text_source, "../descriptions/measure4.txt"))
        self.measure_info_5.clicked.connect(partial(self.change_text_source, "../descriptions/measure5.txt"))



    def plot(self):
        data = pd.read_csv('data.csv')
        ids = data['Responder_id']
        lang_resposnes = data['LanguagesWorkedWith']

        language_counter = Counter()

        for lang in lang_resposnes:
            language_counter.update(lang.split(';'))

        languages, popularity = map(list, zip(*language_counter.most_common(15)))

        plt.barh(languages, popularity)
        plt.title('Languages Popularity')
        plt.ylabel('Languages')
        plt.xlabel('Number of people who use')
        ''' plot some random stuff '''
        #data = [random.random() for i in range(25)]
        ax = self.canvas.figure.add_subplot(111)
        ax.barh(languages, popularity)
        self.canvas.draw()


    def change_text_source(self, path):
        self.textBrowser.setSource(QUrl(path))


    def button_action(self):
        btn1 = self.data1.isChecked()
        btn2 = self.data2.isChecked()
        btn3 = self.data3.isChecked()
        btn4 = self.data4.isChecked()
        btn5 = self.data5.isChecked()

        btn6 = self.algorithm1.isChecked()
        btn7 = self.algorithm2.isChecked()
        btn8 = self.algorithm3.isChecked()
        btn9 = self.algorithm4.isChecked()
        btn10 = self.algorithm5.isChecked()

        btn11 = self.measure1.isChecked()
        btn12 = self.measure2.isChecked()
        btn13 = self.measure3.isChecked()
        btn14 = self.measure4.isChecked()
        btn15 = self.measure5.isChecked()

        self.send_buttons([[btn1, btn2, btn3, btn4, btn5], [btn6, btn7, btn8, btn9, btn10], [btn11, btn12, btn13, btn14, btn15]])

    def send_buttons(self, buttons_list):
        datasets.dataset_function(buttons_list)

if __name__ == '__main__':
    import sys
    from PyQt5 import QtGui

    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

# class MainWindow(QMainWindow):
#
#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__(*args, **kwargs)
#
#         self.setWindowTitle("My Awesome App")
#         self.setGeometry(50, 50, 500, 300)
#         self.setWindowIcon(QIcon('bear.jpg'))
#
#         extractAction = QAction("GO SOMEWHERE I DONT KNOW WHERE", self)
#         extractAction.setShortcut("Ctrl+Q")
#         extractAction.setStatusTip("Leave This App")
#         extractAction.triggered.connect(self.close_application)
#         self.statusBar()
#         mainMenu = self.menuBar()
#         fileMenu = mainMenu.addMenu('File')
#         fileMenu.addAction(extractAction)
#
#         self.home()
#
#     def home(self):
#         btn = QPushButton("Quit", self)
#         btn.clicked.connect(self.close_application)
#         btn.resize(100,100)
#         btn.move(100,100)
#
#         checkBox = QCheckBox('Powiększ okno', self)
#         checkBox.stateChanged.connect(self.enlargeWindow)
#         checkBox.move(100,25)
#
#         self.progress = QProgressBar(self)
#         self.progress.setGeometry(200,200, 250, 50)
#
#         self.btn = QPushButton("Download", self)
#         self.btn.move(200,120)
#         self.btn.clicked.connect(self.download)
#
#
#         self.show()
#
#     def download(self):
#         self.completed = 0
#         while self.completed < 100:
#             self.completed += 0.0001
#             self.progress.setValue(self.completed)
#
#     def enlargeWindow(self, state):
#         if state == Qt.Checked:
#             self.setGeometry(50,50, 1000,600)
#         else:
#             self.setGeometry(50,50, 500,300)
#
#     def close_application(self):
#         choice = QMessageBox.question(self, 'ZAMYKANIE PROGRAMU', 'ARE YOU SURE BRO', QMessageBox.Yes | QMessageBox.No)
#         if choice == QMessageBox.Yes:
#             sys.exit()
#         else:
#             pass
#
#     def closeEvent(self, event):
#         event.ignore()
#         self.close_application()
#
#
#
# app = QApplication(sys.argv)
#
# window = MainWindow()
# window.show()
#
# app.exec_()
