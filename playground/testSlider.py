import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QSlider, QWidget
from PyQt6.QtCore import QTimer, Qt

class SliderPlayer(QMainWindow):
    """
    A class with a slider that can be played and paused.
    Every 100ms, the timer updates and the slider value increments by 1.
    """
    def __init__(self):
        super().__init__()
        self.initUI()

        # Timer Setup
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_slider)
        self.timer_interval = 100  # Update every 100 ms
        self.isPlaying = False

        # Setup for demonstration
        self.current_value = 0


    def initUI(self):
        """
        Initializes the base UI
        """
        # Main Widget and Layout
        self.setWindowTitle('Slider Player')
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.initSlider()

        # Play/Pause Button Setup
        self.playButton = QPushButton('Play', self)
        self.playButton.clicked.connect(self.toggle_play)
        self.layout.addWidget(self.playButton)

        # Set window size and show
        self.resize(300, 100)
        self.show()

    def initSlider(self):
        """
        Initialize the slider with min/max values and its connected signals for
        sliderMoved and valueChanged.
        """
        # Slider Setup
        self.slider = QSlider(Qt.Orientation.Horizontal)  
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)  # Example max value
        self.layout.addWidget(self.slider)

        # Connect slider signals
        self.slider.sliderMoved.connect(self.slider_moved)  # User is dragging the slider
        self.slider.valueChanged.connect(self.slider_value_changed)  # Slider value changed


    def toggle_play(self):
        """Toggle between playing and pausing the timer."""
        if self.isPlaying:
            self.playButton.setText('Play')
            self.timer.stop()
            self.isPlaying = False
        else:
            self.playButton.setText('Pause')
            self.timer.start(self.timer_interval)
            self.isPlaying = True
    

    def update_slider(self):
        """Called every timer_interval ms to update the slider value."""
        # Increment the current value and update the slider
        self.current_value += 1
        if self.current_value > self.slider.maximum():
            self.current_value = 0  # Reset to 0 if it exceeds the maximum
        self.slider.setValue(self.current_value)

    def slider_moved(self, new_position):
        """Called when the slider is being dragged"""
        # Update current_value to match the slider's position when dragged
        self.current_value = new_position

    def slider_value_changed(self, new_value):
        """
        Called anytime the slider value changes. 
        Includes both while dragging and when in 'play' mode.
        """
        # Ensure that current_value matches the slider value when changed
        self.current_value = new_value


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SliderPlayer()
    sys.exit(app.exec())
