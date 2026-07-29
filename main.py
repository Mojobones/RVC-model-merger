import json
import os
import re
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from MergeModels import probe_checkpoint, convert_to_number
from utils.ModelMerger import ModelMergerRequest, MergeElement
from utils.RVCModelMerger import RVCModelMerger

try:  # optional - the app works without it, just with no drag-and-drop
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except Exception:
    HAS_DND = False

MIN_ROWS = 2
OUTPUT_DIR = "merges"
PRESET_FILE = "presets.json"
INVALID_NAME_CHARS = r'<>:"/\|?*'

VOCODER_SUFFIX = {"RefineGAN": "-rfg", "HiFi-GAN": "-hfg"}
VOCODER_COLOR = {"RefineGAN": "#6554c0", "HiFi-GAN": "#0f7b6c"}

# Palette
BG = "#f4f5f7"
CARD = "#ffffff"
BORDER = "#dfe1e6"
TEXT = "#172b4d"
MUTED = "#6b778c"
ACCENT = "#2d7ff9"
ACCENT_DK = "#1c60c9"
DANGER = "#c9372c"

FONT = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 9, "bold")
FONT_TITLE = ("Segoe UI", 13, "bold")


def normalized_sr(value):
    """
    Sample rate is stored inconsistently across checkpoints - 32000, '32000' and
    '32k' all occur in the wild. so normalise before displaying or comparing.
    """
    try:
        return convert_to_number(value)
    except (TypeError, ValueError):
        return None


def show_success(root, message, folder):
    """
    Confirmation with OK as the primary action.
    """
    win = tk.Toplevel(root)
    win.title("Merge complete")
    win.configure(bg=BG)
    win.resizable(False, False)
    win.transient(root)

    tk.Label(win, text=message, bg=BG, fg=TEXT, font=FONT,
             justify="left", wraplength=380).pack(padx=20, pady=(18, 14))

    buttons = tk.Frame(win, bg=BG)
    buttons.pack(padx=20, pady=(0, 16), fill="x")

    def dismiss():
        win.grab_release()
        win.destroy()

    ttk.Button(buttons, text="Open merges folder",
               command=lambda: open_in_file_manager(folder)).pack(side="left")
    ok = ttk.Button(buttons, text="OK", style="Accent.TButton", command=dismiss)
    ok.pack(side="right")

    win.bind("<Return>", lambda e: dismiss())
    win.bind("<Escape>", lambda e: dismiss())
    win.protocol("WM_DELETE_WINDOW", dismiss)

    # Centre on the parent window.
    win.update_idletasks()
    x = root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2
    y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 3
    win.geometry(f"+{max(0, x)}+{max(0, y)}")

    win.grab_set()
    ok.focus_set()
    root.wait_window(win)


def open_in_file_manager(path):
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class ModelRow:
    """One model slot: path + browse + strength + detected model info."""

    def __init__(self, app, parent):
        self.app = app
        self.strength = tk.IntVar(value=50)       # source of truth (drives the spinbox)
        self._scale_var = tk.DoubleVar(value=50)  # ttk.Scale is float-valued
        self._syncing = False
        self.info = dict(arch="unknown", sr=None, version=None, f0=None)

        self.frame = tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                              highlightthickness=1, bd=0)
        self.frame.pack(fill="x", padx=10, pady=4)
        self.frame.columnconfigure(2, weight=1)

        self.index_label = tk.Label(self.frame, text="1", width=2, bg=CARD, fg=MUTED, font=FONT_BOLD)
        self.index_label.grid(row=0, column=0, padx=(10, 2), pady=10)

        moves = tk.Frame(self.frame, bg=CARD)
        moves.grid(row=0, column=1, padx=(0, 6))
        self.up_btn = tk.Button(moves, text="▲", width=2, bd=0, bg=CARD, fg=MUTED,
                                font=("Segoe UI", 7), cursor="hand2",
                                command=lambda: self.app.move_row(self, -1))
        self.up_btn.pack()
        self.down_btn = tk.Button(moves, text="▼", width=2, bd=0, bg=CARD, fg=MUTED,
                                  font=("Segoe UI", 7), cursor="hand2",
                                  command=lambda: self.app.move_row(self, 1))
        self.down_btn.pack()

        self.entry = ttk.Entry(self.frame, font=FONT)
        self.entry.grid(row=0, column=2, sticky="ew", padx=(0, 6), pady=10)

        ttk.Button(self.frame, text="Browse", width=9,
                   command=self.browse).grid(row=0, column=3, padx=(0, 8), pady=10)

        self.badge = tk.Label(self.frame, text="", bg=CARD, fg=MUTED,
                              font=FONT_BOLD, width=20, anchor="w")
        self.badge.grid(row=0, column=4, padx=(0, 6), pady=10)

        self.scale = ttk.Scale(self.frame, from_=1, to=100, orient="horizontal",
                               length=150, style="Row.Horizontal.TScale",
                               variable=self._scale_var, command=self._on_scale)
        self.scale.grid(row=0, column=5, pady=10)

        self.spin = ttk.Spinbox(self.frame, from_=1, to=100, width=5,
                                textvariable=self.strength, font=FONT,
                                command=self._on_spin)
        self.spin.grid(row=0, column=6, padx=(8, 2), pady=10)
        self.strength.trace_add("write", lambda *_: self._on_spin())

        # Strengths are relative, so show what each one actually works out as.
        self.share = tk.Label(self.frame, text="", bg=CARD, fg=MUTED,
                              font=FONT, width=5, anchor="e")
        self.share.grid(row=0, column=7, padx=(0, 4), pady=10)

        self.delete_btn = tk.Button(self.frame, text="✕", width=3, bd=0, bg=CARD,
                                    fg=MUTED, activeforeground=DANGER, cursor="hand2",
                                    font=FONT_BOLD, command=self.delete)
        self.delete_btn.grid(row=0, column=8, padx=(0, 8), pady=10)

        # Pick up paths that were typed or pasted rather than browsed to.
        self.entry.bind("<FocusOut>", lambda e: self.refresh_info())
        self.entry.bind("<Return>", lambda e: self.refresh_info())

        if HAS_DND:
            self.entry.drop_target_register(DND_FILES)
            self.entry.dnd_bind("<<Drop>>", self._on_drop)

    # ---------- model info ----------

    def _on_drop(self, event):
        paths = self.app.parse_dropped(event.data)
        if paths:
            self.app.fill_from(self, paths)
        return "break"

    def set_path(self, path):
        self.entry.delete(0, "end")
        self.entry.insert(0, path)
        self.refresh_info()

    def refresh_info(self):
        """Re-read the checkpoint's metadata and update the badge."""
        path = self.get_path()
        new_info = probe_checkpoint(path) if path else dict(arch="unknown", sr=None, version=None, f0=None)
        changed = new_info != self.info
        self.info = new_info

        arch = self.info["arch"]
        if not path:
            self.badge.config(text="", fg=MUTED)
        elif arch in VOCODER_SUFFIX:
            bits = [arch]
            rate = normalized_sr(self.info["sr"])
            if rate:
                bits.append(f"{rate // 1000}k")
            if self.info["version"]:
                bits.append(str(self.info["version"]))
            if self.info["f0"] is False:
                bits.append("no-f0")
            self.badge.config(text=" · ".join(bits), fg=VOCODER_COLOR[arch])
        else:
            self.badge.config(text="unreadable", fg=DANGER)

        if changed:
            self.app.update_status()
        return changed

    @property
    def arch(self):
        return self.info["arch"]

    # ---------- strength ----------

    def _on_scale(self, _value=None):
        """Slider moved -> round to an int and push into the spinbox."""
        if self._syncing:
            return
        self._syncing = True
        try:
            self.strength.set(int(round(self._scale_var.get())))
        finally:
            self._syncing = False
        self.app.update_status()

    def _on_spin(self):
        """Number typed/stepped -> move the slider to match."""
        if self._syncing:
            return
        try:
            value = int(self.strength.get())
        except (tk.TclError, ValueError):
            return  # mid-edit / non-numeric, leave the slider alone
        self._syncing = True
        try:
            self._scale_var.set(max(1, min(100, value)))
        finally:
            self._syncing = False
        self.app.update_status()

    # ---------- misc ----------

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth")],
                                          initialdir=self.app.last_dir or None)
        if path:
            self.app.last_dir = os.path.dirname(path)
            self.set_path(path)
            self.app.update_status()

    def delete(self):
        self.app.delete_row(self)

    def get_strength(self) -> int:
        """Spinbox lets the user type anything; fall back rather than raising."""
        try:
            value = int(self.strength.get())
        except (tk.TclError, ValueError):
            return 0
        return max(0, min(100, value))

    def get_path(self) -> str:
        return self.entry.get().strip()

    def set_share(self, text):
        self.share.config(text=text)

    def destroy(self):
        self.frame.destroy()


class MergerApp:
    def __init__(self, root):
        self.root = root
        self.rows = []
        self.last_dir = ""
        self.presets = {}
        self.name_var = tk.StringVar()
        self.preset_var = tk.StringVar()
        self.encoder_only = tk.BooleanVar(value=False)
        self._name_manual = False   # user typed their own name; stop auto-filling
        self._setting_name = False  # guard so programmatic writes aren't seen as edits

        root.title("RVC Model Merger")
        root.configure(bg=BG)
        root.geometry("1080x560")
        root.minsize(940, 400)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=FONT, padding=(10, 4))
        style.configure("TEntry", fieldbackground="white", padding=4)
        style.configure("Accent.TButton", font=FONT_BOLD, padding=(18, 6),
                        background=ACCENT, foreground="white", borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", ACCENT_DK), ("disabled", "#a5adba")])
        # Sits on the white row card, so it needs a visibly darker trough.
        style.configure("Row.Horizontal.TScale", background=CARD,
                        troughcolor="#d8dce1", borderwidth=0)

        self.load_presets()
        self._build_header()
        self._build_scroll_area()
        # Packed bottom-up: footer first so it sits lowest, name bar above it.
        self._build_footer()
        self._build_name_bar()

        for _ in range(MIN_ROWS):
            self.add_row()

    # ---------- layout ----------

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(header, text="RVC Model Merger", bg=BG, fg=TEXT,
                 font=FONT_TITLE).pack(side="left")
        hint = "Drag .pth files in, or Browse." if HAS_DND else "Strengths are relative."
        tk.Label(header, text=hint, bg=BG, fg=MUTED, font=FONT).pack(side="left", padx=12)

        ttk.Button(header, text="Delete", width=7,
                   command=self.delete_preset).pack(side="right")
        ttk.Button(header, text="Save", width=6,
                   command=self.save_preset).pack(side="right", padx=4)
        self.preset_combo = ttk.Combobox(header, textvariable=self.preset_var,
                                         state="readonly", width=22, font=FONT)
        self.preset_combo.pack(side="right", padx=(0, 4))
        self.preset_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_preset())
        tk.Label(header, text="Preset", bg=BG, fg=MUTED, font=FONT).pack(side="right", padx=6)
        self._refresh_preset_list()

    def _build_scroll_area(self):
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True, padx=6)

        self.canvas = tk.Canvas(container, bg=BG, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_yview)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.rows_frame = tk.Frame(self.canvas, bg=BG)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")

        # Keep the scrollable region in step with the row list, and make the inner
        # frame span the canvas width so rows stretch instead of hugging the left.
        self.rows_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self.canvas_window, width=e.width))

        # Bind once globally and decide by pointer position. Using <Enter>/<Leave>
        # on the canvas does not work here: moving the pointer onto a child row
        # fires <Leave> on the canvas, which would kill wheel scrolling over
        # exactly the area the user is most likely to be pointing at.
        self.root.bind_all("<MouseWheel>", self._on_wheel)

        if HAS_DND:
            # Dropping onto empty space appends rather than replacing a slot.
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind("<<Drop>>", self._on_canvas_drop)

    def _build_name_bar(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", side="bottom", padx=16, pady=(8, 2))

        tk.Label(bar, text="Save as", bg=BG, fg=TEXT, font=FONT_BOLD).pack(side="left")
        tk.Label(bar, text=f"{OUTPUT_DIR}\\", bg=BG, fg=MUTED, font=FONT).pack(side="left", padx=(8, 0))

        self.name_entry = ttk.Entry(bar, textvariable=self.name_var, font=FONT)
        self.name_entry.pack(side="left", fill="x", expand=True, padx=2)
        tk.Label(bar, text=".pth", bg=BG, fg=MUTED, font=FONT).pack(side="left", padx=(2, 8))

        self.reset_name_btn = ttk.Button(bar, text="Auto", width=6, command=self.reset_name)
        self.reset_name_btn.pack(side="left")

        self.name_var.trace_add("write", self._on_name_edit)

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", side="bottom", padx=16, pady=12)

        ttk.Button(footer, text="+  Add Slot", command=self.add_row).pack(side="left")
        ttk.Button(footer, text="Open Folder",
                   command=lambda: open_in_file_manager(OUTPUT_DIR)).pack(side="left", padx=6)

        self.merge_btn = ttk.Button(footer, text="Merge Models",
                                    style="Accent.TButton", command=self.merge_models)
        self.merge_btn.pack(side="right")

        self.encoder_check = tk.Checkbutton(
            footer, text="Blend encoder only", variable=self.encoder_only,
            command=self.update_status, bg=BG, fg=TEXT, font=FONT,
            activebackground=BG, selectcolor=CARD, bd=0, highlightthickness=0)
        self.encoder_check.pack(side="right", padx=10)

        self.status = tk.Label(footer, text="", bg=BG, fg=MUTED, font=FONT, anchor="w")
        self.status.pack(side="left", padx=14, fill="x", expand=True)

    # ---------- scrolling ----------

    def _pointer_over_canvas(self):
        x, y = self.root.winfo_pointerxy()
        cx, cy = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
        return (cx <= x < cx + self.canvas.winfo_width()
                and cy <= y < cy + self.canvas.winfo_height())

    def _on_wheel(self, event):
        if not self._pointer_over_canvas():
            return
        first, last = self.canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return  # everything already visible - don't let the view drift
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _sync_scrollbar(self):
        """Re-evaluate whether a scrollbar is needed after the row list changes.

        yscrollcommand does not reliably re-fire when the content shrinks but the
        visible fraction stays pinned at the top, so nudge it by hand.
        """
        self.canvas.update_idletasks()
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=(0, 0, bbox[2], bbox[3]))
        self._on_yview(*self.canvas.yview())

    def _on_yview(self, first, last):
        """Drive the scrollbar, and hide it entirely when there is nothing to scroll."""
        self.scrollbar.set(first, last)
        fits = float(first) <= 0.0 and float(last) >= 1.0
        if fits and self.scrollbar.winfo_ismapped():
            self.scrollbar.pack_forget()
        elif not fits and not self.scrollbar.winfo_ismapped():
            self.scrollbar.pack(side="right", fill="y")

    # ---------- drag and drop ----------

    def parse_dropped(self, data):
        """Tk hands back a brace-quoted list; keep only .pth files that exist."""
        try:
            candidates = self.root.tk.splitlist(data)
        except tk.TclError:
            candidates = [data]
        return [p for p in candidates if p.lower().endswith(".pth") and os.path.isfile(p)]

    def fill_from(self, start_row, paths):
        """Drop onto a row: fill it and the rows after it, adding slots as needed."""
        index = self.rows.index(start_row)
        for offset, path in enumerate(paths):
            position = index + offset
            while position >= len(self.rows):
                self.add_row()
            self.rows[position].set_path(path)
        self.last_dir = os.path.dirname(paths[-1])
        self.update_status()

    def _on_canvas_drop(self, event):
        paths = self.parse_dropped(event.data)
        if not paths:
            return "break"
        empty = next((r for r in self.rows if not r.get_path()), None)
        self.fill_from(empty or self._append_row(), paths)
        return "break"

    def _append_row(self):
        self.add_row()
        return self.rows[-1]

    # ---------- presets ----------

    def load_presets(self):
        try:
            with open(PRESET_FILE, encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                self.presets = loaded
        except (OSError, json.JSONDecodeError):
            self.presets = {}

    def _write_presets(self):
        try:
            with open(PRESET_FILE, "w", encoding="utf-8") as handle:
                json.dump(self.presets, handle, indent=2)
            return True
        except OSError as exc:
            messagebox.showinfo("Error", f"Could not save presets:\n\n{exc}")
            return False

    def _refresh_preset_list(self):
        names = sorted(self.presets)
        self.preset_combo.config(values=names)
        if self.preset_var.get() not in names:
            self.preset_var.set("")

    def save_preset(self):
        entries = [{"path": r.get_path(), "strength": r.get_strength()}
                   for r in self.rows if r.get_path()]
        if len(entries) < 2:
            messagebox.showinfo("Error", "Select at least 2 models before saving a preset.")
            return
        name = simpledialog.askstring("Save preset", "Preset name:",
                                      initialvalue=self.preset_var.get() or "")
        if not name:
            return
        name = name.strip()
        if name in self.presets and not messagebox.askyesno(
                "Overwrite?", f'A preset named "{name}" already exists.\n\nReplace it?'):
            return
        self.presets[name] = entries
        if self._write_presets():
            self._refresh_preset_list()
            self.preset_var.set(name)
            self.set_status(f'Preset "{name}" saved.')

    def delete_preset(self):
        name = self.preset_var.get()
        if not name or name not in self.presets:
            messagebox.showinfo("Error", "Choose a preset to delete first.")
            return
        if not messagebox.askyesno("Delete preset?", f'Delete the preset "{name}"?'):
            return
        del self.presets[name]
        if self._write_presets():
            self._refresh_preset_list()
            self.set_status(f'Preset "{name}" deleted.')

    def apply_preset(self):
        name = self.preset_var.get()
        entries = self.presets.get(name)
        if not entries:
            return
        while len(self.rows) > max(MIN_ROWS, len(entries)):
            self.rows.pop().destroy()
        while len(self.rows) < len(entries):
            self.add_row()
        for row, entry in zip(self.rows, entries):
            row.strength.set(entry.get("strength", 50))
            row.set_path(entry.get("path", ""))
        for row in self.rows[len(entries):]:
            row.set_path("")
        self._name_manual = False
        self.refresh()
        missing = [e["path"] for e in entries if e.get("path") and not os.path.isfile(e["path"])]
        if missing:
            messagebox.showinfo(
                "Missing files",
                "This preset refers to files that no longer exist:\n\n" + "\n".join(missing))

    # ---------- rows ----------

    def add_row(self):
        self.rows.append(ModelRow(self, self.rows_frame))
        self.refresh()
        self.root.after_idle(lambda: self.canvas.yview_moveto(1.0))

    def delete_row(self, row):
        if len(self.rows) <= MIN_ROWS:
            self.set_status(f"At least {MIN_ROWS} models are needed for a merge.")
            return
        row.destroy()
        self.rows.remove(row)
        self.refresh()

    def move_row(self, row, delta):
        index = self.rows.index(row)
        target = index + delta
        if not 0 <= target < len(self.rows):
            return
        self.rows[index], self.rows[target] = self.rows[target], self.rows[index]
        for existing in self.rows:          # pack() has no reorder, so re-pack in order
            existing.frame.pack_forget()
        for existing in self.rows:
            existing.frame.pack(fill="x", padx=10, pady=4)
        self.refresh()

    def refresh(self):
        last = len(self.rows) - 1
        for i, row in enumerate(self.rows):
            row.index_label.config(text=str(i + 1))
            can_delete = len(self.rows) > MIN_ROWS
            row.delete_btn.config(state="normal" if can_delete else "disabled",
                                  fg=MUTED if can_delete else "#c1c7d0")
            row.up_btn.config(state="normal" if i > 0 else "disabled",
                              fg=MUTED if i > 0 else "#c1c7d0")
            row.down_btn.config(state="normal" if i < last else "disabled",
                                fg=MUTED if i < last else "#c1c7d0")
        self.update_status()
        self.root.after_idle(self._sync_scrollbar)

    # ---------- name / status ----------

    def _on_name_edit(self, *_):
        if self._setting_name:
            return
        self._name_manual = True
        self.update_status()

    def _set_name(self, text):
        self._setting_name = True
        try:
            self.name_var.set(text)
        finally:
            self._setting_name = False

    def reset_name(self):
        """Drop a manual name and go back to the generated one."""
        self._name_manual = False
        self.update_status()

    def active_rows(self):
        return [r for r in self.rows if r.get_path() and r.get_strength() > 0]

    def merged_vocoder(self):
        """The shared architecture of the selected models, or None if not agreed."""
        archs = {r.arch for r in self.active_rows()} - {"unknown"}
        return archs.pop() if len(archs) == 1 else None

    def build_merged_name(self, entries):
        # Unchanged from the original naming scheme (first 4 chars + strength),
        # with an optional vocoder suffix appended.
        name = ""
        for path, strength in entries:
            match = re.search(r'[^\\/]+(?=\.pth$)', path)
            if match:
                name += match.group()[:4] + str(strength)
        return name + VOCODER_SUFFIX.get(self.merged_vocoder(), "")

    def update_status(self):
        active = self.active_rows()
        total = sum(r.get_strength() for r in active)
        for row in self.rows:
            row.set_share(f"{100.0 * row.get_strength() / total:.0f}%"
                          if row in active and total else "")

        entries = [(r.get_path(), r.get_strength()) for r in active]
        auto_name = self.build_merged_name(entries) if len(entries) >= 2 else ""
        if not self._name_manual:
            self._set_name(auto_name)
        self.reset_name_btn.config(state="normal" if self._name_manual else "disabled")

        archs = {r.arch for r in active} - {"unknown"}
        rates = {normalized_sr(r.info["sr"]) for r in active} - {None}
        versions = {str(r.info["version"]) for r in active if r.info["version"]}
        partial = self.encoder_only.get()

        if len(archs) > 1:
            self.set_status("Mixed vocoders selected — these models cannot be merged.")
        elif len(versions) > 1:
            self.set_status(f"Mixed RVC versions ({', '.join(sorted(versions))}) "
                            "— these models cannot be merged.")
        elif len(entries) < 2:
            self.set_status("Select at least 2 models with a strength above 0.")
        elif len(rates) > 1 and not partial:
            rate_list = ", ".join(f"{s // 1000}k" for s in sorted(rates))
            self.set_status(f"Mixed sample rates ({rate_list}) — tick 'Blend encoder only' to "
                            f"combine them, with the {min(rates) // 1000}k model in slot 1.")
        elif partial:
            base_rate = normalized_sr(active[0].info["sr"]) if active else None
            rate = f"{base_rate // 1000}k" if base_rate else "?"
            # Blending disturbs the base model's latents proportionally more at higher
            # rates, so the lowest-rate model makes the most forgiving base.
            if rates and base_rate and base_rate > min(rates):
                self.set_status(f"Partial blend — output {rate}. Tip: move the "
                                f"{min(rates) // 1000}k model to slot 1 for better results.")
            else:
                self.set_status(f"Partial blend — slot 1 supplies the decoder, output is {rate}. "
                                "Only the encoder is mixed.")
        else:
            vocoder = self.merged_vocoder()
            self.set_status(f"{len(entries)} models" + (f"  ·  {vocoder}" if vocoder else ""))

    def output_name(self):
        """Whatever will actually be written, with unusable characters stripped."""
        raw = self.name_var.get().strip()
        return "".join(c for c in raw if c not in INVALID_NAME_CHARS).strip(" .")

    def set_status(self, text):
        self.status.config(text=text)

    # ---------- merge ----------

    def merge_models(self):
        rows = [(r, r.get_path(), r.get_strength()) for r in self.rows]

        missing = [f"Slot {i}" for i, (_, p, s) in enumerate(rows, 1) if s > 0 and not p]
        if missing:
            messagebox.showinfo("Error", "No file selected for: " + ", ".join(missing))
            return

        not_found = [p for _, p, s in rows if s > 0 and p and not os.path.isfile(p)]
        if not_found:
            messagebox.showinfo("Error", "These files could not be found:\n\n" + "\n".join(not_found))
            return

        entries = [(p, s) for _, p, s in rows if p and s > 0]
        if len(entries) < 2:
            messagebox.showinfo("Error", "Please provide 2 or more models with a strength above 0.")
            return

        active = self.active_rows()
        rates = {normalized_sr(r.info["sr"]) for r in active} - {None}
        if len(rates) > 1 and not self.encoder_only.get():
            rate_list = ", ".join(f"{s // 1000}k" for s in sorted(rates))
            messagebox.showinfo(
                "Mixed sample rates",
                f"These models use different sample rates ({rate_list}), so their "
                "decoders cannot be blended.\n\n"
                "Tick 'Blend encoder only' to combine them anyway. That mixes only the "
                "sample-rate-independent part of the network and takes the decoder "
                "(and the output sample rate) from slot 1.\n\n"
                f"For the best results put the lowest sample rate ({min(rates) // 1000}k) "
                "in slot 1 — a higher-rate base is disturbed more by the blend and tends "
                "to sound degraded.")
            return

        merged_name = self.output_name()
        if not merged_name:
            messagebox.showinfo("Error", "Please enter a name for the merged model.")
            self.name_entry.focus_set()
            return

        target = os.path.join(OUTPUT_DIR, merged_name + ".pth")
        if os.path.exists(target) and not messagebox.askyesno(
                "Overwrite?",
                f"{merged_name}.pth already exists in {OUTPUT_DIR}\\.\n\nReplace it?"):
            return

        files = [MergeElement(p, s) for p, s in entries]

        self.merge_btn.config(state="disabled")
        self.set_status("Merging…")
        self.root.update_idletasks()
        try:
            request = ModelMergerRequest(command="merge", files=files, mergedName=merged_name,
                                         encoderOnly=self.encoder_only.get())
            _, success = RVCModelMerger().merge_models(request)
            if success:
                self.set_status(f"Saved  {OUTPUT_DIR}\\{merged_name}.pth")
                show_success(self.root,
                             f"Merged model saved as\n{OUTPUT_DIR}\\{merged_name}.pth",
                             OUTPUT_DIR)
            else:
                self.set_status("Merge failed — see the error above.")
        except Exception as exc:  # keep the window alive instead of dying to a traceback
            self.set_status("Merge failed.")
            messagebox.showinfo("Error", f"Merge failed:\n\n{type(exc).__name__}: {exc}")
        finally:
            self.merge_btn.config(state="normal")


if __name__ == "__main__":
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    MergerApp(root)
    root.mainloop()
