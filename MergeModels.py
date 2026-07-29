from typing import Dict, Any, List, Tuple
import io
import os
import pickle
import zipfile
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


# ---------- Helper utilities for Option A ----------

def _pad_shape(shape: Tuple[int, ...], length: int) -> Tuple[int, ...]:
    """Pad a shape with leading 1s to a given length."""
    if len(shape) >= length:
        return shape
    return (1,) * (length - len(shape)) + tuple(shape)


def _broadcastable(target: Tuple[int, ...], src: Tuple[int, ...]) -> bool:
    """
    Check PyTorch-style broadcastability between shapes:
    - Right-aligned
    - Each pair of dims must be equal or one must be 1
    """
    lt, ls = len(target), len(src)
    L = max(lt, ls)
    tt = _pad_shape(target, L)
    ss = _pad_shape(src, L)
    for dt, ds in zip(reversed(tt), reversed(ss)):
        if dt != ds and dt != 1 and ds != 1:
            return False
    return True


def _target_shape_for_key(key: str, model_weights: List[Dict[str, torch.Tensor]]) -> Tuple[int, ...]:
    """
    Choose a target shape for a key by taking the max along each aligned dimension
    across all available tensors for that key. This makes order-independent merges possible.
    """
    shapes: List[Tuple[int, ...]] = []
    for mw in model_weights:
        t = mw.get(key)
        if t is not None and isinstance(t, torch.Tensor):
            shapes.append(tuple(t.shape))
    if not shapes:
        return tuple()
    max_len = max(len(s) for s in shapes)
    padded = [_pad_shape(s, max_len) for s in shapes]
    target_dims: List[int] = []
    for dims in zip(*padded):
        target_dims.append(max(dims))
    return tuple(target_dims)


def _infer_common_dtype(model_weights: List[Dict[str, torch.Tensor]], key: str) -> torch.dtype:
    """Pick a reasonable dtype for a given key (prefer the first tensor's dtype)."""
    for mw in model_weights:
        t = mw.get(key)
        if isinstance(t, torch.Tensor):
            return t.dtype
    return torch.float32


def detect_vocoder_arch(weights: Dict[str, torch.Tensor]) -> str:
    """
    Identify which decoder architecture a checkpoint's weights belong to.

    Read from the tensor names rather than from checkpoint metadata, because the
    `vocoder` field is frequently missing (older checkpoints, and anything merged
    before this was fixed) while the decoder layer names are always definitive.
    """
    keys = list(weights.keys())
    if any(("upsample_conv_blocks" in k) or ("downsample_blocks" in k) or ("mel_conv" in k) for k in keys):
        return "RefineGAN"
    if any(k.startswith("dec.ups.") or ("noise_convs" in k) or ("resblocks" in k) for k in keys):
        return "HiFi-GAN"
    return "unknown"


class _Stub:
    """Placeholder standing in for anything we deliberately refuse to construct."""

    def __init__(self, *args, **kwargs):
        pass


_SAFE_CLASSES = {("collections", "OrderedDict"): OrderedDict}


class _MetadataUnpickler(pickle.Unpickler):
    """
    Reads a checkpoint's structure without building (or importing) anything real.

    Every class except the plain containers we allow is swapped for a stub, and
    tensor storages are dropped via persistent_load, so no torch code runs and no
    tensor bytes are touched. Scalar metadata (sr / version / f0 / vocoder) and the
    state-dict *key names* survive intact, which is all the UI needs.
    """

    def find_class(self, module, name):
        return _SAFE_CLASSES.get((module, name), _Stub)

    def persistent_load(self, pid):
        return None


EMPTY_PROBE = {"arch": "unknown", "sr": None, "version": None, "f0": None}


def probe_checkpoint(path: str) -> Dict[str, Any]:
    """
    Read a checkpoint's architecture and metadata without loading its tensors.

    A .pth is a zip archive whose small `data.pkl` entry holds the whole object
    graph minus the tensor payloads. Parsing just that entry costs ~3ms and a few
    tens of KB, versus reading the full 50MB file, which keeps the UI responsive
    when a model is picked. Falls back to a real load for legacy non-zip .pth files.
    """
    if not path or not os.path.isfile(path):
        return dict(EMPTY_PROBE)

    data = None
    try:
        with zipfile.ZipFile(path) as archive:
            pkl = next((n for n in archive.namelist() if n.endswith("data.pkl")), None)
            if pkl is not None:
                data = _MetadataUnpickler(io.BytesIO(archive.read(pkl))).load()
    except (zipfile.BadZipFile, OSError, pickle.UnpicklingError, EOFError, AttributeError):
        data = None

    if data is None:
        try:
            data = torch.load(path, map_location="cpu")
        except Exception:
            return dict(EMPTY_PROBE)

    if not isinstance(data, dict):
        return dict(EMPTY_PROBE)

    weights = data.get("weight") or data.get("model") or {}
    arch = detect_vocoder_arch(weights) if isinstance(weights, dict) else "unknown"
    if arch == "unknown" and isinstance(data.get("vocoder"), str):
        arch = data["vocoder"]

    def scalar(key):
        value = data.get(key)
        return value if isinstance(value, (int, float, str, bool)) else None

    return {"arch": arch, "sr": scalar("sr"), "version": scalar("version"), "f0": scalar("f0")}


def probe_vocoder(path: str) -> str:
    """Architecture only - kept as the narrow entry point used by the merge guard."""
    return probe_checkpoint(path)["arch"]


# ---------------------------------------------------


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
        return None, False

    weights_wrapped = []  # list of {"weight": {...}} like your extract() returns
    flat_weights = []     # list of dict[str, Tensor] i.e., just the "weight" sub-dicts
    alphas = []
    model_paths = []      # parallel to flat_weights, for readable error messages
    merge_model_sample_rate = None
    state_dict = None  # keep last loaded's metadata like your original

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

        # Ensure all models share the same sample rate
        if convert_to_number(model_sample_rate) != convert_to_number(merge_model_sample_rate):
            messagebox.showinfo(
                "Error",
                f"Please ensure all models are the same sample rate!\n "
                f"First model in set was {merge_model_sample_rate} but then "
                f"received {model_sample_rate}"
            )
            return None, False

        weights_wrapped.append(weight)
        flat_weights.append(weight["weight"] if "weight" in weight else weight)
        alphas.append(f.strength)
        model_paths.append(filename)

    if len(flat_weights) < 2:
        messagebox.showinfo("Error", "Please provide 2 or more models to merge with non-zero Strength.")
        return None, False

    # Ensure all models share the same decoder architecture. A RefineGAN decoder and
    # a HiFi-GAN decoder have almost entirely different layer names, so merging them
    # would not blend anything: the union below would just staple both sets of
    # weights into one file, each scaled down by its own strength, and whichever
    # architecture the loader picked would see the other one's tensors as garbage.
    archs = [detect_vocoder_arch(mw) for mw in flat_weights]
    known = {a for a in archs if a != "unknown"}
    if len(known) > 1:
        detail = "\n".join(
            f"  {os.path.basename(p)}: {a}" for p, a in zip(model_paths, archs)
        )
        messagebox.showinfo(
            "Error",
            "Cannot merge models that use different vocoders.\n"
            "These models do not share a decoder architecture:\n\n"
            f"{detail}\n\n"
            "Merge RefineGAN models only with other RefineGAN models, and "
            "HiFi-GAN models only with other HiFi-GAN models."
        )
        return None, False
    merged_arch = next(iter(known)) if known else "unknown"
    print(f"Detected vocoder architecture: {merged_arch}")

    # Normalize alphas (sum to 1.0)
    alpha_sum = float(sum(alphas))
    if alpha_sum == 0.0:
        messagebox.showinfo("Error", "Sum of Strength values is 0.")
        return None, False
    alphas = [float(a) / alpha_sum for a in alphas]

    # Build a union of all parameter keys (safer than requiring exact key match)
    all_keys = set()
    for mw in flat_weights:
        all_keys.update(mw.keys())
    all_keys = sorted(all_keys)

    merged: Dict[str, Any] = OrderedDict()
    merged["weight"] = {}
    skipped_log = []  # collect (key, out_shape, bad_shape) we skipped due to incompatibility
    used_as_fallback = []  # keys where we had to fall back to the first model's tensor

    for key in all_keys:
        # Compute a target shape that can host broadcasted additions
        tgt_shape = _target_shape_for_key(key, flat_weights)
        if len(tgt_shape) == 0:
            # key not found in any model (shouldn't happen since it's from union), skip
            continue

        dtype = _infer_common_dtype(flat_weights, key)
        out = torch.zeros(tgt_shape, dtype=dtype)

        contributed = 0
        alpha_used = 0.0
        for alpha, mw in zip(alphas, flat_weights):
            if key not in mw:
                continue
            t = mw[key]
            if not isinstance(t, torch.Tensor):
                continue

            # Safe, non-in-place add with broadcasting if possible
            if _broadcastable(out.shape, tuple(t.shape)):
                # ensure dtype compatibility
                out = out + t.to(dtype) * alpha
                alpha_used += alpha
                contributed += 1
            else:
                skipped_log.append((key, tuple(out.shape), tuple(t.shape)))

        # The alphas were normalised across *every* model, so when only some of them
        # actually supplied this tensor the contributing alphas no longer sum to 1
        # and the result would be silently attenuated (e.g. a tensor only one model
        # of two owns would come out at 40% strength). Rescale by the weight that
        # really contributed. No-op for the normal case where every model has the key.
        if contributed and out.is_floating_point() and abs(alpha_used - 1.0) > 1e-6 and alpha_used > 0:
            out = out / alpha_used

        # If nothing could contribute (e.g., every tensor was incompatible), keep base tensor from the first model that has it
        if contributed == 0:
            for mw in flat_weights:
                if key in mw and isinstance(mw[key], torch.Tensor):
                    out = mw[key]
                    used_as_fallback.append(key)
                    break

        merged["weight"][key] = out

    # Carry over metadata from the last loaded state_dict. Copy every non-weight
    # field rather than a hardcoded list: a hardcoded list silently dropped
    # `vocoder`, which is what tells a loader to build a RefineGAN decoder instead
    # of a HiFi-GAN one. Without it, merged RefineGAN models fail to load anywhere
    # that trusts that field. Copying generically also preserves `speakers_id`,
    # `embedder_model`, author/name fields, and anything future formats add.
    for meta_key, meta_value in state_dict.items():
        if meta_key in ("weight", "model"):
            continue  # tensors, not metadata - already merged above
        merged[meta_key] = meta_value

    # Keys older consumers of this merger expect to always exist, even if the
    # source checkpoints did not carry them.
    for legacy_key in ("params", "version", "info", "embedder_name", "embedder_output_layer"):
        merged.setdefault(legacy_key, None)

    # Fall back to the architecture detected from the tensors themselves when the
    # source checkpoints predate the `vocoder` field.
    if merged.get("vocoder") is None and merged_arch != "unknown":
        merged["vocoder"] = merged_arch

    # Logging to help you see what happened
    if skipped_log:
        print("Merge notice: skipped non-broadcastable tensors for these keys (target vs source shapes):")
        for k, s_out, s_in in skipped_log[:50]:
            print(f"  - {k}: {s_out} vs {s_in}")
        if len(skipped_log) > 50:
            print(f"  ... and {len(skipped_log) - 50} more")
    if used_as_fallback:
        print(f"Merge notice: {len(used_as_fallback)} keys could not be merged; kept tensor from first available model for those keys.")

    print("Wrote metadata.")
    return merged, True
