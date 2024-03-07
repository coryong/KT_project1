from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os

current_path = os.path.dirname(os.path.abspath(__file__))


form_class_thirdpage = uic.loadUiType(os.path.join(current_path,"ui/Qt_designer_thirdpage.ui"))[0]

class ThirdPage(QMainWindow, form_class_thirdpage):
    def __init__(self, log_list = None):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Cheat Block")
        self.setGeometry(150, 150, 870, 990)

        self.log_list = log_list

        self.pushButton.clicked.connect(self.close)

        self.display_logs()
    
    def display_logs(self):
        if self.log_list:
            log_texts = []
            for log in self.log_list:
                log_text = f"탐지내용: {log[0]}\n시험시간: {log[1]}초 후\n탐지시간: {log[2]}\n"
                log_texts.append(log_text)

            model = QStringListModel(log_texts)
            self.listView.setModel(model)



if __name__ == "__main__":
    app = QApplication([])
    window = ThirdPage()
    window.show()
    app.exec_()