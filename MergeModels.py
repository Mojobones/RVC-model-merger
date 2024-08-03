from typing import Dict, Any
import os
from collections import OrderedDict
from tkinter import messagebox

import torch
from utils.ModelMerger import ModelMergerRequest

def convert_to_number(s):
    if isinstance(s, str):
        s = s.lower().replace(' ', '')
        if s.endswith('k'):
            return int(float(s[:-1]) * 1000)
        elif s.endswith('m'):
            return int(float(s[:-1]) * 1000000)
    return int(s)

def merge_model(request: ModelMergerRequest):
    global state_dict

    def extract(ckpt: Dict[str, Any]):
        a = ckpt["model"]
        opt: Dict[str, Any] = OrderedDict()

        opt["weight"] = {}
        for loc_key in a.keys():
            if "enc_q" in loc_key:
                continue
            opt["weight"][loc_key] = a[loc_key]
        return opt

    def load_weight(path: str):
        print(f"Loading {path}...")
        loc_state_dict = torch.load(path, map_location="cpu")
        if "model" in loc_state_dict:
            loc_weight = extract(loc_state_dict)
        else:
            loc_weight = loc_state_dict["weight"]
        return loc_weight, loc_state_dict

    files = request.files
    if len(files) == 0:
        messagebox.showinfo("Error", f"Please provide 2 or more models to merge")

    weights = []
    alphas = []
    merge_model_sample_rate = None

    for f in files:
        strength = f.strength
        if strength == 0:
            print("Skipping " + f.modelPath + " as Strength value is 0.")
            continue

        filename = f.modelPath

        weight, state_dict = load_weight(filename)

        model_sample_rate = state_dict["sr"]

        if merge_model_sample_rate is None:

            merge_model_sample_rate = model_sample_rate

        # If we hit this, the user didn't provide all the same sample rate models
        if convert_to_number(model_sample_rate) != convert_to_number(merge_model_sample_rate):
            messagebox.showinfo("Error", f"Please ensure all models are the same sample rate!\n "
                                         f"First model in set was {merge_model_sample_rate} but then "
                                         f"received {model_sample_rate}")
            return None, False

        weights.append(weight)
        alphas.append(f.strength)

    alphas = [x / sum(alphas) for x in alphas]

    for weight in weights:
        if sorted(list(weight.keys())) != sorted(list(weights[0].keys())):
            raise RuntimeError("Failed to merge models.")

    merged: Dict[str, Any] = OrderedDict()
    merged["weight"] = {}
    for key in weights[0].keys():
        merged["weight"][key] = 0
        for i, weight in enumerate(weights):
            merged["weight"][key] += weight[key] * alphas[i]

    merged["config"] = state_dict["config"]
    merged["params"] = state_dict["params"] if "params" in state_dict else None
    merged["version"] = state_dict["version"] if "version" in state_dict else None
    merged["sr"] = state_dict["sr"]
    merged["f0"] = state_dict["f0"]
    merged["info"] = state_dict["info"] if "info" in state_dict else None
    merged["embedder_name"] = state_dict["embedder_name"] if "embedder_name" in state_dict else None
    merged["embedder_output_layer"] = state_dict[
        "embedder_output_layer"] if "embedder_output_layer" in state_dict else None
    print("Wrote metadata.")
    return merged, True
