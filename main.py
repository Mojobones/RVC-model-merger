import torch
import tkinter as tk
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox, Scale
import re

from utils.ModelMerger import ModelMergerRequest, MergeElement
from utils.RVCModelMerger import RVCModelMerger

root = Tk()
root.title("RVC Model Merger")
model_rows = []


def select_file(entry):
    filepath = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth")])
    entry.delete(0, "end")
    entry.insert(0, filepath)


def merge_models():
    """Begins the merging for the selected models
    Retrieves data about each of the user-entered models nad creates a merge request
    that will merge the weights
    """
    files = []
    merged_name = ""

    # loop through and grab each path and strength value
    for frame, entry, slider, button in model_rows:
        path = entry.get()
        strength = slider.get()

        # For the saved merge, we're going to use a combination of
        # the various names truncated to 4 chars
        match = re.search(r'[^\\/]+(?=\.pth$)', path)

        if match:
            filename = match.group()[:4]
            merged_name += (filename + str(strength))

        # append the MergeElement to our Files that will get processed
        files.append(MergeElement(path, strength))

    # create merge request and send it
    request = ModelMergerRequest(command="merge", files=files, mergedName=merged_name)
    rvc_merger = RVCModelMerger()
    model, success = rvc_merger.merge_models(request)

    if success:
        messagebox.showinfo("Success", f"Merged model saved as {merged_name}")


def add_row():
    """Adds a row to the main window
    Adds a row to the main window with entry controls for a model to merge
    """
    frame = tk.Frame(root)
    frame.pack(fill='x', padx=5, pady=5)

    entry = tk.Entry(frame, width=50)
    entry.pack(side='left', padx=5, pady=5)

    browse_button = tk.Button(frame, text='Browse', command=lambda: select_file(entry))
    browse_button.pack(side='left', padx=5)

    slider = tk.Scale(frame,
                      from_=1, to=100,
                      orient='horizontal',
                      label='Strength',
                      showvalue=False)
    slider.pack(side='left', padx=5)

    slider_label = tk.Label(frame, text=str(slider.get()))
    slider_label.pack(side='left', padx=5)

    def update_label(value):
        slider_label.config(text=str(value))

    def on_label_click(event):
        input_value = tk.simpledialog.askinteger("Input", "Enter a value:", minvalue=1, maxvalue=100)
        if input_value is not None:
            slider.set(input_value)

    slider_label.bind("<Button-1>", on_label_click)
    slider.config(command=update_label)

    button = tk.Button(frame, text='Delete', command=lambda: delete_row(frame))
    button.pack(side='right', padx=5, pady=5)

    model_rows.append((frame, entry, slider, button))


def delete_row(frame):
    global model_rows
    for row in model_rows:
        if row[0] == frame:
            model_rows.remove(row)
            break
    for widget in frame.winfo_children():
        widget.destroy()
    frame.pack_forget()

    # Update the layout and resize the window
    root.update_idletasks()


# Create a separate frame for the "Merge Models" button
button_frame = tk.Frame(root)
button_frame.pack(side='bottom', fill='x', pady=10)

tk.Button(root, text="Add Merge Slot", command=add_row).pack(pady=10)

# Add default rows
add_row()
add_row()

tk.Button(button_frame, text="Merge Models", command=merge_models).pack()

root.mainloop()
