import sys
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QPushButton,
    QWidget,
    QFileDialog,
    qApp,
    QMessageBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QRadioButton,
    QPlainTextEdit,
    QComboBox,
    QListWidget,
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import logging

import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import momanalysis.main as mom
from momanalysis.utility import logger
import subprocess
import matplotlib.pyplot as plt
plt.switch_backend('qt5agg')

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.files = []
        self.fluorescence_stack = None
        self.integrated_fluo = False
        self.output_file = None
        self.currently_selected_file = None
        self.batch = False

    def initUI(self):

        self.statusBar().showMessage("No File Added")
        self.mw = MainWidget(self)
        self.setCentralWidget(self.mw)
        self.setGeometry(300, 300, 800, 300)
        self.show()


    def findfile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,
            "Select input file", "",
            "All Files (*);;Python Files (*.py)",
            options=options)
        if fileName:
            self.statusBar().showMessage("File added: " + fileName)
            #self.files.append(fileName)
            #self.files = fileName if isinstance(fileName, list) else [fileName,]
            self.mw.startbutton.setEnabled(True)
            self.mw.outputlabel.setEnabled(True)
            self.mw.brightfield_box.setChecked(True)
            self.mw.removeselectedfiles.setEnabled(True)
            self.currently_selected_file = fileName
            self.update_files_added()

    def exitmomanalysis(self):
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes |
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            qApp.quit()

    def fluoresc_stack(self, state):
        self.fluorescence_stack = 1 if state else None

    def integr_fluo(self, state):
        self.integrated_fluo = state

    #def brightfield_or_fluorescence(self, state):
    #    pass

    def outputcheckbox(self, state):
        if state == Qt.Checked:
            self.mw.output.setEnabled(True)
        else:
            self.mw.output.setEnabled(False)
            self.mw.output.setText("Optional: Specify output filename.  N.B. it will be followed by a timestamp")

    def remove_files(self):

        self.statusBar().showMessage("File removed. Please add a new file")
        for item in self.mw.selectedfiles.selectedItems():
            self.mw.selectedfiles.takeItem(self.mw.selectedfiles.row(item))
            filetoberemoved = item.text()
            self.files.remove(filetoberemoved)
            if len(self.files) < 1:
                self.mw.removeselectedfiles.setEnabled(False)
                self.statusBar().showMessage("Add a file before starting analysis")
                self.mw.startbutton.setEnabled(False)
                self.mw.comb_fluorescence.setEnabled(False)
                self.mw.comb_fluorescence.setChecked(False)
                self.mw.seper_fluorescence.setEnabled(False)
                self.mw.seper_fluorescence.setChecked(False)
                self.mw.brightfield_box.setChecked(True)


    def manually_entered_file(self):
        self.currently_selected_file = self.mw.filesadded.text()
        self.update_files_added()
        self.mw.filesadded.setText(self.mw.added_files_text)
        self.mw.removeselectedfiles.setEnabled(True)
        self.mw.startbutton.setEnabled(True)

    def update_files_added(self):
        if self.currently_selected_file is not None:
            self.files.append(self.currently_selected_file)
        self.mw.selectedfiles.clear()
        for file in self.files:
            self.mw.selectedfiles.addItem(str(file))
        self.currently_selected_file = None

    def batch_or_not(self,state):
        self.batch = state == Qt.Checked


class MainWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dir_name = []
        self.info_level = "INFO"
        self.setAcceptDrops(True)

        self.infolevel_dict = logging._levelToName
        self.initUI()

    def initUI(self):
        self.added_files_text = "write the full file path, manually choose or drag and drop the files"

        self.addfilesbutton = QPushButton("Choose Files")
        self.addfilesbutton.clicked.connect(self.parent().findfile)

        self.selectedfiles = QListWidget(self)
        #self.selectedfiles.setReadOnly(True)

        self.exitbutton = QPushButton("Exit")
        self.exitbutton.clicked.connect(self.parent().exitmomanalysis)

        self.startbutton = QPushButton("Start Analysis")
        self.startbutton.clicked.connect(self.start_analysis)
        self.startbutton.setEnabled(False)

        self.seeresults = QPushButton("Open results folder")
        self.seeresults.setEnabled(False)
        self.seeresults.clicked.connect(lambda: launch_file_explorer(self.dir_name))

        self.output = QLineEdit("Optional: Specify output filename.  N.B. it will be followed by a timestamp")
        self.output.setEnabled(False)
        self.outputlabel = QCheckBox("Use own name for output File:")
        self.outputlabel.setEnabled(False)
        self.outputlabel.stateChanged.connect(self.parent().outputcheckbox)

        self.addfiles = QLabel("Input file:")
        self.welcome = QLabel("Welcome to Momanalysis")
        self.filesadded = QLineEdit(self.added_files_text)
        self.filesadded.returnPressed.connect(self.parent().manually_entered_file)

        self.removeselectedfiles = QPushButton("Remove selected files")
        self.removeselectedfiles.setEnabled(False)
        self.removeselectedfiles.clicked.connect(self.parent().remove_files)

        self.analysisoptions = QLabel("Select options")

        ###
        ### Start file-mode options (i.e. Brightfield only, combined, seperate)
        ###
        self.brightfield_box = QRadioButton("Brightfield Only")
        self.brightfield_box.setToolTip("<b>Default.</b><br>Images are only brightfield and no fluorescence analysis is required.")
        self.brightfield_box.setChecked(True)
        #self.brightfield_box.toggled.connect(self.parent().brightfield_or_fluorescence)

        self.comb_fluorescence = QRadioButton("Combined Fluorescence")
        self.comb_fluorescence.setToolTip("Select if your stack contains alternating brightfield and fluorescent images")
        self.comb_fluorescence.toggled.connect(self.parent().integr_fluo)
        #self.comb_fluorescence.setEnabled(False)

        self.seper_fluorescence = QRadioButton("Seperate Fluorescence")
        self.seper_fluorescence.setToolTip("Select if you have a stack of brightfield images and a separate stack of matching fluorescent images")
        self.seper_fluorescence.toggled.connect(self.parent().fluoresc_stack)
        #self.seper_fluorescence.setEnabled(False)
        ###
        ### End of file-mode options
        ###

        self.set_debug = QLabel("Select info level")
        self.set_debug.setToolTip("Select the level of logging information to display:<br><b>Not Set</b> = all information <br><b>Critical</b> = only major errors.<br><br>Default is set to <b>Info</b>.")
        self.debug_info_level = QComboBox(self)
        self.debug_info_level.addItems(self.infolevel_dict.values())
        self.debug_info_level.activated[str].connect(self.set_info_level)

        self.batchcheckbox = QCheckBox("Batch run")
        self.batchcheckbox.setEnabled(True)
        self.batchcheckbox.setChecked(False)
        self.batchcheckbox.stateChanged.connect(self.parent().batch_or_not)

        self.logwidget = QPlainTextEdit(self)
        self.logwidget.setReadOnly(True)


        font = QFont()
        font.setBold(True)
        font.setPointSize(8)
        self.welcome.setFont(font)

        #edit layout

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(self.welcome,0,0)
        grid.addWidget(self.addfilesbutton,1,5)
        grid.addWidget(self.selectedfiles,2,1)
        grid.addWidget(self.removeselectedfiles,2,5)
        grid.addWidget(self.filesadded,1,1)
        grid.addWidget(self.addfiles,1,0)
        grid.addWidget(self.exitbutton,14,6)
        grid.addWidget(self.outputlabel,9,0)
        grid.addWidget(self.output,9,1)
        grid.addWidget(self.analysisoptions,3,0)
        grid.addWidget(self.brightfield_box,4,0)
        grid.addWidget(self.comb_fluorescence,5,0)
        grid.addWidget(self.seper_fluorescence,6,0)
        grid.addWidget(self.startbutton, 10, 1)
        grid.addWidget(self.seeresults, 11, 1)
        grid.addWidget(self.set_debug,12,0)
        grid.addWidget(self.debug_info_level,12,1)
        grid.addWidget(self.logwidget,13,1)
        grid.addWidget(self.batchcheckbox,7,0)


        #set layout
        self.setLayout(grid)

        self.thread = AnalysisThread(self.parent(), self)
        self.thread.finished_analysis.connect(self.updateUi)
        self.thread.log_message.connect(self.addLogMessage)

    def set_info_level(self, str):
        self.info_level = str #self.infolevel_dict[str]
        return self.info_level
        #print(self.infolevel_dict[str], flush = True)


    def start_analysis(self):
        if self.thread.isRunning():
            self.parent().output_file = None
            #self.setEnabled(True)
            for child in self.children():
                if hasattr(child, "setEnabled"):
                    child.setEnabled(True)
            self.thread.terminate()
            self.startbutton.setText('Start Analysis')
            logger.critical("Run aborted by user")
            self.parent().statusBar().showMessage("Run aborted by user. Please add a new file and start again")
            self.output.setText("Optional: Specify output filename.  N.B. it will be followed by a timestamp")
            self.filesadded.setText(self.added_files_text)
            self.parent().files = []
            self.parent().update_files_added()
            #self.removeselectedfiles.setEnabled(False)
            #self.setAcceptDrops(True)
        else:
            self.startbutton.setText('Stop Analysis')

            #self.setEnabled(False)
            for child in self.children():
                if hasattr(child, "setEnabled"):
                    child.setEnabled(False)
            self.startbutton.setEnabled(True)

            self.parent().statusBar().showMessage("Running analysis")
            if self.outputlabel.isChecked():
                self.parent().output_file = str(self.output.text())
            else:
                self.parent().output_file = None
            #self.addfilesbutton.setEnabled(False)
            #self.filesadded.setEnabled(False)
            #self.comb_fluorescence.setEnabled(False)
            #self.seper_fluorescence.setEnabled(False)
            #self.seeresults.setEnabled(False)
            #self.outputlabel.setEnabled(False)
            #self.setAcceptDrops(False)
            self.thread.start()

    def addLogMessage(self, msg):
        self.logwidget.appendPlainText(msg)

    def updateUi(self, dir_name):
        #self.setEnabled(True)
        for child in self.children():
            if hasattr(child, "setEnabled"):
                child.setEnabled(True)
        self.startbutton.setText('Start Analysis')
        self.seeresults.setEnabled(True)
        self.addfilesbutton.setEnabled(True)
        self.parent().statusBar().showMessage("Analysis Finished. Click to see your files or please add a new file")
        self.dir_name = dir_name
        self.filesadded.setEnabled(True)
        self.filesadded.setText("Finished. Type new file path, manually select or drag file")


    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if os.path.isfile(url.toLocalFile()) or os.path.isdir(url.toLocalFile()):
                path = url.toLocalFile()
                #self.parent().files = path if isinstance(path, list) else [path,]
                #self.filesadded.setText(path)
                self.parent().currently_selected_file = path
                self.startbutton.setEnabled(True)
                self.outputlabel.setEnabled(True)
                self.parent().statusBar().showMessage("File added: " + path)
                self.removeselectedfiles.setEnabled(True)
                self.brightfield_box.setChecked(True)
                self.parent().update_files_added()



class AnalysisThread(QThread):

    finished_analysis = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, parent=None, parent2 = None):
        QThread.__init__(self)
        self.parent = parent
        self.mainWidget = parent2


    def run(self):
        output = self.parent.output_file
        fluo = self.parent.fluorescence_stack
        fluoresc = self.parent.integrated_fluo
        inputfile = self.parent.files
        self.loghandler = QLogHandler(self)
        """This changes the format of the GUI messages"""
        #self.loghandler.setFormatter(logger.formatter)
        logger.addHandler(self.loghandler)
        #using the numeric values (20 is default = Info)
        logger.setLevel(self.mainWidget.info_level)

        if self.parent.batch == True:
            dir_name = mom.batch(
                inputfile,
                output = output,
                fluo =fluo,
                fluoresc= fluoresc,
                batch=True)
        else:
            dir_name = mom.run_analysis_pipeline(
                inputfile,
                output = output,
                fluo =fluo,
                fluoresc= fluoresc,
            )
        self.finished_analysis.emit(dir_name)


def launch_file_explorer(path):
    # Try cross-platform file-explorer opening...
    # Courtesy of: http://stackoverflow.com/a/1795849/537098
    if sys.platform=='win32':
        subprocess.Popen(['start', path], shell= True)
    elif sys.platform=='darwin':
        subprocess.Popen(['open', path])
    else:
        try:
            subprocess.Popen(['xdg-open', path])
        except OSError:
            # er, think of something else to try
            # xdg-open *should* be supported by recent Gnome, KDE, Xfce
            QMessageBox.critical(self,
                "Oops",
                "\n".join(["Couldn't launch the file explorer, sorry!"
                           "Manually open %s in your favourite file manager"%path])
            )

class QLogHandler(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.parent=parent

    def emit(self, record):
        msg = self.format(record)
        self.parent.log_message.emit(msg)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.setWindowTitle("Momanalysis")
    ex.show()
    app.exec_()
