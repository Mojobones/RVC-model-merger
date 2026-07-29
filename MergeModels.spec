# -*- mode: python ; coding: utf-8 -*-
#
# Build:  pyinstaller MergeModels.spec --clean --noconfirm
#
# Produces dist/RVC-Model-Merger/ - a self-contained folder. Nothing from the
# working tree is bundled except the source modules below: no merges/, no
# presets.json, no model files.

from PyInstaller.utils.hooks import collect_all

# tkinterdnd2 ships native Tcl libraries that must be collected explicitly,
# otherwise drag-and-drop silently fails in the frozen build.
tkdnd_datas, tkdnd_binaries, tkdnd_hidden = collect_all("tkinterdnd2")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=tkdnd_binaries,
    datas=tkdnd_datas,
    hiddenimports=tkdnd_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Not used by the merger; excluded to keep the download smaller.
    excludes=[
        "matplotlib", "scipy", "pandas", "PIL", "IPython", "notebook",
        "pytest", "torchvision", "torchaudio", "tensorboard",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RVC-Model-Merger",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX is known to corrupt some torch DLLs
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RVC-Model-Merger",
)
