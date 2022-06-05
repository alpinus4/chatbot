from tkinter import *
import config as c
from chat import Chat


class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        self.chat = Chat()
        self.chat.load_data()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=c.BG_COLOR)

        head_label = Label(self.window, bg=c.BG_GRAY, fg=c.TEXT_COLOR, text="Welcome", font=c.FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=c.BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        self.text_widget = Text(self.window, width=20, height=2, bg=c.BG_COLOR, fg=c.TEXT_COLOR, font=c.FONT, padx=5,
                                pady=5)  # display 20 characters in line, we have two lines
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)  # whenever we scroll yview will change

        bottom_label = Label(self.window, bg=c.BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        self.msg_entry = Entry(bottom_label, bg=c.ENTRY_BG, fg=c.TEXT_COLOR, font=c.FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()  # at the beginning focus on message that we can write message at once
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        send_button = Button(bottom_label, text="Send", font=c.FONT_BOLD, width=20, bg=c.BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:  # if we press enter without text init
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"{c.BOT_NAME}: {self.chat.get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)


def main():
    app = ChatApplication()
    app.run()


if __name__ == "__main__":
    main()
