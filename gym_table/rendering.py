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