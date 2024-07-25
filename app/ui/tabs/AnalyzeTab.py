from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel


class AnalyzeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        
        self.label = QLabel("Analyze violin audio here...", self)
        self.layout.addWidget(self.label)

    def status_message(self):
        return "Analyzing recording..."