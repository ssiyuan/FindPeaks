import sys
from PyQt6.QtWidgets import QApplication, QMainWindow

import FindPeaks

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = FindPeaks.Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    app.exec()