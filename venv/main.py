import sys
from PyQt5.uic import loadUiType
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import stat_tests
import csv
import datasets
from functools import partial
import threading
import os

#import pliku stworzonego w QtDesigner
Ui_MainWindow, QMainWindow = loadUiType('prototyp.ui')
file_path = "test.csv"
default_path = "default.csv"

class Worker(QRunnable):

    def __init__(self, buttons_list, menu):
        super(Worker, self).__init__()
        self.bl = buttons_list
        self.menu = menu


    @pyqtSlot()
    def run(self):
        print("Thread start")
        datasets.dataset_function(self.bl, self.menu)
        self.menu.graphs_list = self.menu.generate_graphs(file_path, self.bl[1], self.bl[2])
        self.menu.current_graph = 0
        if len(self.menu.graphs_list) > 0:
            self.menu.plot.setPixmap(QtGui.QPixmap(self.menu.graphs_list[self.menu.current_graph]))
            self.menu.name.setText(self.menu.graphs_names[self.menu.current_graph])
            self.menu.next.setEnabled(True)
        print("Thread complete")


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        self.threadpool = QThreadPool()
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.button_action) #wywołanie buttona "test"
        self.textBrowser.setSource(QUrl("../descriptions/welcome.txt")) #załadowanie tekstu o textBrowser
        self.pushButton_2.clicked.connect(self.button2_action)
        self.graphs_list = []
        self.current_graph = 0
        self.next.setEnabled(False)
        self.previous.setEnabled(False)
        self.next.clicked.connect(self.next_action)
        self.previous.clicked.connect(self.previous_action)

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

    def closeEvent(self, event):
        for i in self.graphs_list:
            os.remove(i)
        if os.path.exists(file_path):
            os.remove(file_path)


    def change_text_source(self, path):
        self.textBrowser.setSource(QUrl(path))


    def button_action(self):
        box1 = self.data1.isChecked()
        box2 = self.data2.isChecked()
        box3 = self.data3.isChecked()
        box4 = self.data4.isChecked()
        box5 = self.data5.isChecked()

        box6 = self.algorithm1.isChecked()
        box7 = self.algorithm2.isChecked()
        box8 = self.algorithm3.isChecked()
        box9 = self.algorithm4.isChecked()
        box10 = self.algorithm5.isChecked()

        box11 = self.measure1.isChecked()
        box12 = self.measure2.isChecked()
        box13 = self.measure3.isChecked()
        box14 = self.measure4.isChecked()
        box15 = self.measure5.isChecked()

        box16 = self.split1.isChecked()
        box17 = self.split2.isChecked()
        box18 = self.split3.isChecked()
        box19 = self.split4.isChecked()
        box20 = self.split5.isChecked()

        data_checkboxes = [box1, box2, box3, box4, box5]
        algorithms_checkboxes = [box6, box7, box8, box9, box10]
        measures_checkboxes = [box11, box12, box13, box14, box15]
        ratio_checkboxes = [box16, box17, box18, box19, box20]

        if sum(data_checkboxes) > 0 and sum(algorithms_checkboxes) > 2 and sum(measures_checkboxes) > 0 and sum(ratio_checkboxes) > 0:
            self.textBrowser.setPlainText(
                "System is working... please wait")
            self.send_buttons([data_checkboxes, algorithms_checkboxes, measures_checkboxes, ratio_checkboxes])

        else:
            self.textBrowser.setPlainText("You have to choose at least one database, one split ratio, one measure and three models")


    #wysłanie checkboxów do drugiego modułu
    def send_buttons(self, buttons_list):
        #wyłączamy pushbuttony
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)

        #tworzymy nowy wątek
        self.worker = Worker(buttons_list, self)
        self.threadpool.start(self.worker)


    #akcja dla przycisku rysującego wykresy dla domyślnych danych
    def button2_action(self):
        self.graphs_list = self.generate_graphs(default_path, [True, True, True, True, True], [True, True, True, True, True])
        self.current_graph = 0
        if len(self.graphs_list) > 0:
            self.name.setText(self.graphs_names[self.current_graph])
            self.plot.setPixmap(QtGui.QPixmap(self.graphs_list[self.current_graph]))
            self.next.setEnabled(True)


    #akcja dla przycisku "next"
    def next_action(self):
        self.current_graph += 1
        self.plot.setPixmap(QtGui.QPixmap(self.graphs_list[self.current_graph]))
        self.name.setText(self.graphs_names[self.current_graph])
        if self.current_graph == len(self.graphs_list) - 1:
            self.next.setEnabled(False)
        if not self.previous.isEnabled():
            self.previous.setEnabled(True)


    #akcja dla przycisku "previous"
    def previous_action(self):
        self.current_graph -= 1
        self.plot.setPixmap(QtGui.QPixmap(self.graphs_list[self.current_graph]))
        self.name.setText(self.graphs_names[self.current_graph])
        if self.current_graph == 0:
            self.previous.setEnabled(False)
        if not self.next.isEnabled():
            self.next.setEnabled(True)


    #funkcja wywołująca funkcję z modułu stat_test i dodająca do nazw wykresów rozszerzenie .png
    def generate_graphs(self, path, algorithms, measures):
        self.graphs_names = stat_tests.statistic_tests(path, algorithms, measures)
        graphs = [i+".png" for i in self.graphs_names]
        return graphs

if __name__ == '__main__':
    import sys
    from PyQt5 import QtGui

    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

