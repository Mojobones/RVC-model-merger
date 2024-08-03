import torch
import tkinter as tk
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox, Scale
from MergeModels import merge_model

from utils.ModelMerger import ModelMergerRequest, MergeElement
from utils.RVCModelMerger import RVCModelMerger

root = Tk()
root.title("RVC Model Merger")
modelRows = []


def select_file(entry):
    filepath = filedialog.askopenfilename()
    entry.delete(0, "end")
    entry.insert(0, filepath)


def merge_models():
    files = []

    for frame, entry, slider, button in modelRows:
        path = entry.get()
        strength = slider.get()

        files.append(MergeElement(path, strength))

    request = ModelMergerRequest(
        command="merge",
        files=files)

    rvc_merger = RVCModelMerger()

    merged_model_path = rvc_merger.merge_models(request)

    messagebox.showinfo("Success", f"Merged model stored at: {merged_model_path}")


def add_row():
    frame = tk.Frame(root)
    frame.pack(fill='x')

    entry = tk.Entry(frame, width=50)
    entry.pack(side='left', padx=5, pady=5)

    browse_button = tk.Button(frame, text='Browse', command=lambda: select_file(entry))
    browse_button.pack(side="left", padx=5, pady=5);

    slider = tk.Scale(frame, from_=1, to=100, orient='horizontal')
    slider.pack(side='left', padx=5, pady=5)

    button = tk.Button(frame, text='Delete', command=lambda: delete_row(frame))
    button.pack(side='right', padx=5, pady=5)

    modelRows.append((frame, entry, slider, button))


def delete_row(frame):
    # Remove all widgets in the row
    for widget in frame.winfo_children():
        widget.destroy()
    frame.pack_forget()  # Remove the frame
    modelRows.remove(frame)  # Remove frame from list


tk.Button(root, text="Add Model", command=add_row).pack(pady=10)
tk.Button(root, text="Merge Models", command=merge_models).pack()

root.mainloop()
