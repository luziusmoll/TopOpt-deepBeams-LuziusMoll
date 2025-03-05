from tkinter import Tk, Label, Entry, Button, Frame, StringVar, messagebox
import json
import os

class GeometryInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Geometry Input")

        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.label = Label(self.frame, text="Enter Geometry Parameters:")
        self.label.grid(row=0, column=0, columnspan=2)

        self.param1_label = Label(self.frame, text="Parameter 1:")
        self.param1_label.grid(row=1, column=0)
        self.param1_var = StringVar()
        self.param1_entry = Entry(self.frame, textvariable=self.param1_var)
        self.param1_entry.grid(row=1, column=1)

        self.param2_label = Label(self.frame, text="Parameter 2:")
        self.param2_label.grid(row=2, column=0)
        self.param2_var = StringVar()
        self.param2_entry = Entry(self.frame, textvariable=self.param2_var)
        self.param2_entry.grid(row=2, column=1)

        self.submit_button = Button(self.frame, text="Submit", command=self.submit)
        self.submit_button.grid(row=3, column=0, columnspan=2)

    def submit(self):
        param1 = self.param1_var.get()
        param2 = self.param2_var.get()

        if not param1 or not param2:
            messagebox.showerror("Input Error", "Please fill in all fields.")
            return

        parameters = {
            "parameter1": param1,
            "parameter2": param2
        }

        self.save_parameters(parameters)
        messagebox.showinfo("Success", "Parameters saved successfully!")

    def save_parameters(self, parameters):
        config_path = os.path.join(os.path.dirname(__file__), '../config/parameters.json')
        with open(config_path, 'w') as json_file:
            json.dump(parameters, json_file)

def main():
    root = Tk()
    gui = GeometryInputGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()