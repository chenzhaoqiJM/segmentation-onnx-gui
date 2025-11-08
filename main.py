import sys
from PyQt5.QtWidgets import QApplication
# from PyQt5.QtGui import QResizeEvent
from sub import SubPage

if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = SubPage()
    mainWindow.show()
    mainWindow.resizeWindow()

    sys.exit(app.exec_())