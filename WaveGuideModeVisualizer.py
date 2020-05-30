from PyQt5 import QtWidgets
import os
import numpy as np
import math as m

from PyQt5.QtWidgets import QDialog, QComboBox, QDoubleSpinBox, QDialogButtonBox, QFormLayout, QMessageBox
from mayavi.mlab import quiver3d, clf, outline, colorbar

os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'
from pyface.qt import QtGui, QtCore
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

## create class for generating data for TE and TM Mode in Rectangular WaveGuide
## Considerations: Most naturally occurring materials are non-magnetic at optical frequencies,
## that is μr is very close to 1,therefore n is approximately √ε

""" Rectangular WaveGuide Class operating in TM_mn Mode """


class RectTM:
    def __init__(self, a, b, om, eps, mu, TMm=1, TMn=1):
        self.m = TMm
        self.n = TMn
        if a > b:
            self.a = a
            self.b = b
        else:
            self.a = b
            self.b = a
        self.om = om
        self.eps = eps
        self.mu = mu
        self.E_0 = 1.
        self.h = self.set_h()
        self.gamma = self.set_gamma()
        self.cutoffFreq = self.cutoffFrequency()

    def modConfig(self, a, b, om, eps, mu, TMm=1, TMn=1):
        self.m = TMm
        self.n = TMn
        if a > b:
            self.a = a
            self.b = b
        else:
            self.a = b
            self.b = a
        self.om = om
        self.eps = eps
        self.mu = mu
        self.E_0 = 1.
        self.h = self.set_h()
        self.gamma = self.set_gamma()
        self.cutoffFreq = self.cutoffFrequency()

    def set_h(self):
        return m.sqrt(m.pow(self.m * m.pi / self.a, 2) + m.pow(self.n * m.pi / self.b, 2))

    def set_gamma(self):
        return m.sqrt(-m.pow(self.h, 2) + (self.om * self.om * self.eps * self.mu))

    def cutoffFrequency(self):
        coeff = 1 / (2 * m.sqrt(self.mu * self.eps))
        return coeff * m.sqrt(m.pow(1. / self.a, 2) + m.pow(1. / self.b, 2))

    def Ez(self, x, y, z, t):
        coeff = m.cos(self.om * t - self.gamma * z)
        cff = coeff * self.E_0
        return cff * m.sin(self.m * m.pi * x / self.a) * m.sin(self.n * m.pi * y / self.b)

    def Hz(self, x, y, z, t):
        return 0.0

    def Ex(self, x, y, z, t):
        coeff = - self.gamma / m.pow(self.h, 2)
        caff = coeff * self.m * m.pi * self.E_0 / self.a
        cff = caff * m.sin(self.om * t - self.gamma * z)
        return cff * m.cos(self.m * m.pi * x / self.a) * m.sin(self.n * m.pi * y / self.b)

    def Ey(self, x, y, z, t):
        coeff = - self.gamma / m.pow(self.h, 2)
        caff = coeff * self.n * m.pi * self.E_0 / self.b
        cff = caff * m.sin(self.om * t - self.gamma * z)
        return cff * m.sin(self.m * m.pi * x / self.a) * m.cos(self.n * m.pi * y / self.b)

    def Hx(self, x, y, z, t):
        coeff = self.om * self.mu / m.pow(self.h, 2)
        caff = coeff * self.n * m.pi * self.E_0 / self.b
        cff = caff * m.sin(self.om * t - self.gamma * z)
        return cff * m.sin(self.m * m.pi * x / self.a) * m.cos(self.n * m.pi * y / self.b)

    def Hy(self, x, y, z, t):
        coeff = self.om * self.mu / m.pow(self.h, 2)
        caff = coeff * self.m * m.pi * self.E_0 / self.a
        cff = caff * m.sin(self.om * t - self.gamma * z)
        return cff * m.cos(self.m * m.pi * x / self.a) * m.sin(self.n * m.pi * y / self.b)


""" Rectangular WaveGuide Class operating in TE_mn Mode """


class RectTE:
    def __init__(self, a, b, om, eps, mu, TEm=1, TEn=1):
        if a > b:
            self.a = a
            self.b = b
        else:
            self.a = b
            self.b = a
        self.m = TEm
        self.n = TEn
        self.om = om
        self.eps = eps
        self.mu = mu
        self.H_0 = 1.
        self.h = self.set_h()
        self.gamma = self.set_gamma()
        self.cutoffFreq = self.cutoffFrequency()

    def modConfig(self, a, b, om, eps, mu, TEm=1, TEn=1):
        self.m = TEm
        self.n = TEn
        if a > b:
            self.a = a
            self.b = b
        else:
            self.a = b
            self.b = a
        self.om = om
        self.eps = eps
        self.mu = mu
        self.H_0 = 1.
        self.h = self.set_h()
        self.gamma = self.set_gamma()
        self.cutoffFreq = self.cutoffFrequency()

    def set_h(self):
        return m.sqrt(m.pow(self.m * m.pi / self.a, 2) + m.pow(self.n * m.pi / self.b, 2))

    def set_gamma(self):
        return m.sqrt(-m.pow(self.h, 2) + (self.om * self.om * self.eps * self.mu))

    def cutoffFrequency(self):
        coeff = 1 / (2 * m.sqrt(self.mu * self.eps))
        return coeff / self.a

    def Hz(self, x, y, z, t):
        coeff = m.cos(self.om * t - self.gamma * z)
        cff = coeff * self.H_0
        return cff * m.cos(self.m * m.pi * x / self.a) * m.cos(self.n * m.pi * y / self.b)

    def Ez(self, x, y, z, t):
        return 0.0

    def Hx(self, x, y, z, t):
        coeff = self.gamma / m.pow(self.h, 2)
        caff = coeff * self.m * m.pi * self.H_0 / self.a
        cff = caff * m.sin(self.om * t - self.gamma * z)
        return cff * m.sin(self.m * m.pi * x / self.a) * m.cos(self.n * m.pi * y / self.b)

    def Hy(self, x, y, z, t):
        coeff = self.gamma / m.pow(self.h, 2)
        caff = coeff * self.n * m.pi * self.H_0 / self.b
        cff = caff * m.sin(self.om * t - self.gamma * z)
        return cff * m.cos(self.m * m.pi * x / self.a) * m.sin(self.n * m.pi * y / self.b)

    def Ex(self, x, y, z, t):
        coeff = self.om * self.mu / m.pow(self.h, 2)
        caff = coeff * self.n * m.pi * self.H_0 / self.b
        cff = caff * m.sin(self.om * t - self.gamma * z)
        return cff * m.cos(self.m * m.pi * x / self.a) * m.sin(self.n * m.pi * y / self.b)

    def Ey(self, x, y, z, t):
        coeff = - self.om * self.mu / m.pow(self.h, 2)
        caff = coeff * self.m * m.pi * self.H_0 / self.a
        cff = caff * m.sin(self.om * t - self.gamma * z)
        return cff * m.sin(self.m * m.pi * x / self.a) * m.cos(self.n * m.pi * y / self.b)

class PopupClass:

    def show_popupTE(self):
        msg = QMessageBox()
        msg.setWindowTitle("Wrong Set of Configuration for TE Mode")
        msg.setText("The Base Mode for Transverse Electric(TE) Mode is (1, 0). "
                    "Since the Mode parameters are invalid, it is automatically set to (1, 0)")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_popupTM(self):
        msg = QMessageBox()
        msg.setWindowTitle("Wrong Set of Configuration for TM Mode")
        msg.setText("The Base Mode for Transverse Magnetic(TM) Mode is (1, 1). "
                    "Since the Mode parameters are invalid, it is automatically set to (1, 1)")
        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


## create Mayavi Widget and show

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    lsto = ['TM Mode', 'Electric Field', 10.0, 0.05, 0.03, 6.0, 1.0, 1.0]

        #self.on_trait_change(self.update_plot, name='lsto')

    @on_trait_change('scene.activated')
    def update_plot(self):
        clf()

        lst = self.lsto
        lst[6] = int(lst[6])
        lst[7] = int(lst[7])

        if lst[4] >= lst[3]:
            tmp = lst[4]
            lst[4] = lst[3]
            lst[3] = tmp

        if 'TE Mode' == lst[0]:
            if lst[6] >= 1 and lst[7] >= 0:
                tmp = 100
            else:
                popup = PopupClass()
                popup.show_popupTE()
                lst[6] = 1
                lst[7] = 0
                lst[5] = 1.0
            field = RectTE(lst[3], lst[4], lst[2] * 1e+9,
                           8.85418782e-12 * (lst[5] ** 2), 1.25663706e-6, lst[6], lst[7])
            if 'Electric Field' == lst[1]:
                func1 = np.vectorize(field.Ex)
                func2 = np.vectorize(field.Ey)
                func3 = np.vectorize(field.Ez)
            else:
                func1 = np.vectorize(field.Hx)
                func2 = np.vectorize(field.Hy)
                func3 = np.vectorize(field.Hz)
        else:
            if lst[6] >= 1 and lst[7] >= 1:
                tmp = 100
            else:
                popup = PopupClass()
                popup.show_popupTM()
                lst[6] = 1
                lst[7] = 1
                lst[5] = 1.0
            field = RectTM(lst[3], lst[4], lst[2] * 1e+9,
                           8.85418782e-12 * (lst[5] ** 2), 1.25663706e-6, lst[6], lst[7])
            if 'Electric Field' == lst[1]:
                func1 = np.vectorize(field.Ex)
                func2 = np.vectorize(field.Ey)
                func3 = np.vectorize(field.Ez)
            else:
                func1 = np.vectorize(field.Hx)
                func2 = np.vectorize(field.Hy)
                func3 = np.vectorize(field.Hz)

        x_lim = lst[3] * 100.
        y_lim = lst[4] * 100.

        X, Y, Z = np.mgrid[0:x_lim:25j, 0:y_lim:25j, 0:5:20j]
        obj = quiver3d(X, Y, Z, func1(X, Y, Z, 1.), func2(X, Y, Z, 1.), func3(X, Y, Z, 1.))

        colorbar()
        #outline()

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False), resizable=True)


class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)



#### PyQt5 GUI ####
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        ## MAIN WINDOW
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(200, 200, 1100, 700)

        ## CENTRAL WIDGET
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        ## GRID LAYOUT
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        ## BUTTONS
        self.button_previous_data = QtWidgets.QPushButton(self.centralwidget)
        self.button_previous_data.setObjectName("button_previous_data")
        self.gridLayout.addWidget(self.button_previous_data, 2, 0, 1, 1)
        self.button_previous_data.clicked.connect(self.on_buttonclick)

        ## Mayavi Widget 1
        container = QtGui.QWidget()
        mayavi_widget = MayaviQWidget(container)
        self.gridLayout.addWidget(mayavi_widget, 1, 0, 1, 1)

        ## SET TEXT
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "WaveGuide Mode Visualizer"))
        self.button_previous_data.setText(_translate("MainWindow", "Change Values"))

    def on_buttonclick(self):
        dialog = InputDialog()
        vis = Visualization()
        if dialog.exec():
            vis.lsto = list(dialog.getInputs())
            vis.update_plot()


class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.zeroth = QComboBox(self)
        self.zeroth.addItems(['TE Mode', 'TM Mode'])
        self.fielD = QComboBox(self)
        self.fielD.addItems(['Electric Field', 'Magnetic Field'])
        self.opFreq = QDoubleSpinBox(self)
        self.len = QDoubleSpinBox(self)
        self.wid = QDoubleSpinBox(self)
        self.second = QDoubleSpinBox(self)
        self.modeM = QDoubleSpinBox(self)
        self.modeN = QDoubleSpinBox(self)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        self.setWindowTitle("Choose Your Configuration")
        layout.addRow("Mode Selector : ", self.zeroth)
        layout.addRow("Which Field to Visualize : ", self.fielD)
        layout.addRow("Operating Frequency(in GHz) : ", self.opFreq)
        layout.addRow("Length of WaveGuide(a) in cm(s) : ", self.len)
        layout.addRow("Width of WaveGuide(b) in cm(s) : ", self.wid)
        layout.addRow("Refractive Index of Dielectric : ", self.second)
        layout.addRow("Mode (m, n) Value of m : ", self.modeM)
        layout.addRow("Mode (m, n) Value of n : ", self.modeN)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return self.zeroth.currentText(), self.fielD.currentText(), self.opFreq.value(), self.len.value() / 100., \
               self.wid.value() / 100., self.second.value(), self.modeM.value(), self.modeN.value()

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())