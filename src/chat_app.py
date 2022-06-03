from tkinter import *
import config as c

class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()
        
    def _setup_mainwindow(self):
        self.window.title("chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=c.BG_COLOR)