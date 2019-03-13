import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPolygon
from PyQt5.QtCore import QPoint, QSize, QRect
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTextEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QFrame
import sys

class Window(QMainWindow):
    """
    Simple application to render window environment
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Table Environment')

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

        # Stores keyboard input
        self.keyName = None

    def closeEvent(self, event):
        """Close an event"""
        self.closed = True

    def setText(self, text):
        """ Enter the mission text"""
        self.missionBox.setPlainText(text)

    def keyPressEvent(self, e):
        """Get keyboard inputs"""
        # Keyboard inputs
        self.keyName = None
        if e.key() == Qt.Key_Left:
            self.keyName = 'left'
        elif e.key() == Qt.Key_Right:
            self.keyName = 'right'
        elif e.key() == Qt.Key_Up:
            self.keyName = 'up'
        elif e.key() == Qt.Key_Down:
            self.keyName = 'down'
        elif e.key() == Qt.Key_Escape:
            self.keyName = 'escape'
            sys.exit(0)
        if self.keyName == None:
            print("Unrecognized key input")
            return


class Renderer:
    def __init__(self, width=128, height=128, ownWindow=True):
        self.width = width
        self.height = height

        self.img = QImage(width, height, QImage.Format_RGB888)

        self.painter = QPainter()

        self.window = None
        if ownWindow:
            self.app = QApplication(sys.argv)
            self.window = Window()
        #sys.exit(self.app.exec_())

    def beginFrame(self):
        self.painter.begin(self.img)
        self.painter.setRenderHint(QPainter.Antialiasing, False)

        # Clear the background
        self.painter.setBrush(QColor(0, 0, 0))
        self.painter.drawRect(0, 0, self.width - 1, self.height - 1)

    def push(self):
        self.painter.save()

    def pop(self):
        self.painter.restore()

    def rotate(self, degrees):
        self.painter.rotate(degrees)

    def translate(self, x, y):
        self.painter.translate(x, y)

    def scale(self, x, y):
        self.painter.scale(x, y)

    def setLineColor(self, r, g, b, a=255):
        self.painter.setPen(QColor(r, g, b, a))

    def setColor(self, r, g, b, a=255):
        self.painter.setBrush(QColor(r, g, b, a))

    def setLineWidth(self, width):
        pen = self.painter.pen()
        pen.setWidthF(width)
        self.painter.setPen(pen)

    def drawLine(self, x0, y0, x1, y1):
        self.painter.drawLine(x0, y0, x1, y1)

    def drawCircle(self, x, y, r):
        center = QPoint(x, y)
        self.painter.drawEllipse(center, r, r)

    def drawPolygon(self, points):
        """Takes a list of points (tuples) as input"""
        points = map(lambda p: QPoint(p[0], p[1]), points)
        self.painter.drawPolygon(QPolygon(points))

    def drawPolyline(self, points):
        """Takes a list of points (tuples) as input"""
        points = map(lambda p: QPoint(p[0], p[1]), points)
        self.painter.drawPolyline(QPolygon(points))

    def fillRect(self, x, y, width, height, r, g, b, a=255):
        self.painter.fillRect(QRect(x, y, width, height), QColor(r, g, b, a))

if __name__ == "__main__":
    while True:
        r = Renderer()
        print("Hello")
