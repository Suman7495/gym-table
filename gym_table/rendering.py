import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPolygon
from PyQt5.QtCore import QPoint, QSize, QRect
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTextEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QFrame

class Window(QMainWindow):
    """
    Simple application to render window environment
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Table Gym Environment')

        # Image label to display the rendering
        self.imgLabel = QLabel()
        self.imgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)


        # Text box for the mission
        self.missionBox = QTextEdit()
        self.missionBox.setReadOnly(True)
        self.missionBox.setMinimumSize(400, 100)

        # Center the image
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.imgLabel)
        hbox.addStretch(1)

        # Arrange widgets vertically
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.missionBox)

        # Create a main widget for the window
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        mainWidget.setLayout(vbox)

        # Show the application window
        self.show()
        self.setFocus()

        self.closed = False

        # Callback for keyboard events
        self.keyDownCb = None

    def closeEvent(self, event):
        """Close an event"""
        self.closed = True

    def setKeyDownCb(self, callback):
        """Read keyboard input"""
        self.keyDownCb = callback

    def setText(self, text):
        """ Enter the mission text"""
        self.missionBox.setPlainText(text)

    def keyPressEvent(self, e):
        if self.keyDownCb == None:
            return

        # Keyboard inputs
        keyName = None
        if e.key() == Qt.Key_Left:
            keyName = 'left'
        elif e.key() == Qt.Key_Right:
            keyname = 'right'
        elif e.key() == Qt.Key_Up:
            keyName = 'up'
        elif e.key() == Qt.Key_Down:
            keyName = 'down'
        elif e.key() == Qt.Key_Escape:
            keyName = 'escape'

        if keyName == None
            return

        self.keyDownCb(keyName)


class Renderer:
    def __init__(self, width, height, ownWindow=False):
        self.width = width
        self.height = height

        self.img = QImage(width, height, QImage.Format_RGB888)

        self.painter = QPainter()

        self.window = None
        if ownWindow:
            self.app = QApplication([])
            self.window = Window()


if __name__ == "__main__":
    r = Renderer(128, 128, True)
