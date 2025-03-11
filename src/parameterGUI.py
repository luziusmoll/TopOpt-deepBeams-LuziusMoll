from tkinter import Tk, Label, Entry, Button, Frame, StringVar, messagebox
import json
import os

class ParameterInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Parameter Input")

        self.frame = Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.label = Label(self.frame, text="Enter Parameters:")
        self.label.grid(row=0, column=0, columnspan=2)

        self.create_parameter_input("Volume Fraction (volfrac):", "0.4", 1)
        self.create_parameter_input("Penalty:", "3", 2)
        self.create_parameter_input("Minimum Density (x_min):", "1e-3", 3)
        self.create_parameter_input("Filter Radius (r_min):", "0.15", 4)
        self.create_parameter_input("Young's Modulus:", "30000", 5)
        self.create_parameter_input("Poisson's Ratio:", "0.15", 6)
        self.create_parameter_input("Maximum Number of Iterations:", "50", 7)
        self.create_parameter_input("Mesh Element Size Factor:", "10", 8)

        self.submit_button = Button(self.frame, text="Submit", command=self.submit)
        self.submit_button.grid(row=9, column=0, columnspan=2)

    def create_parameter_input(self, label_text, default_value, row):
        label = Label(self.frame, text=label_text)
        label.grid(row=row, column=0)
        var = StringVar(value=default_value)
        entry = Entry(self.frame, textvariable=var)
        entry.grid(row=row, column=1)
        setattr(self, f"param{row}_var", var)

    def submit(self):
        parameters = {
            "volfrac": self.param1_var.get(),
            "penalty": self.param2_var.get(),
            "x_min": self.param3_var.get(),
            "r_min": self.param4_var.get(),
            "Youngs_modulus": self.param5_var.get(),
            "Poissons_ratio": self.param6_var.get(),
            "max_iteration": self.param7_var.get(),
            "mesh_el_size": self.param8_var.get()
        }

        if any(not value for value in parameters.values()):
            messagebox.showerror("Input Error", "Please fill in all fields.")
            return

        self.save_parameters(parameters)
        # messagebox.showinfo("Success", "Parameters saved successfully!")
        self.master.destroy()  # Close the dialog box

    def save_parameters(self, parameters):
        config_path = os.path.join(os.path.dirname(__file__), '../config/parameters.json')
        with open(config_path, 'w') as json_file:
            json.dump(parameters, json_file)

def main():
    root = Tk()
    gui = ParameterInputGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()