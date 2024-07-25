from PyQt6.QtWidgets import QStatusBar

class StatusBar(QStatusBar):
    """
    Initialize the status bar.
    @param
        parent: The parent widget.
    """

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.showMessage("Ready")
