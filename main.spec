# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules
import os, struct

def ensure_ico_from_png(png_path: str, ico_path: str):
    try:
        if os.path.exists(ico_path):
            return
        if not os.path.exists(png_path):
            return
        with open(png_path, 'rb') as f:
            png = f.read()
        if png[:8] != b'\x89PNG\r\n\x1a\n':
            return
        width = int.from_bytes(png[16:20], 'big')
        height = int.from_bytes(png[20:24], 'big')
        header = struct.pack('<HHH', 0, 1, 1)
        w8 = width if 0 < width < 256 else 0
        h8 = height if 0 < height < 256 else 0
        entry = struct.pack('<BBBBHHII', w8, h8, 0, 0, 1, 32, len(png), 6 + 16)
        with open(ico_path, 'wb') as f:
            f.write(header)
            f.write(entry)
            f.write(png)
    except Exception:
        pass

ensure_ico_from_png('images/logos/ShimIcon(200x200px).png', 'images/logos/ShimIcon.ico')

hiddenimports = collect_submodules('uvicorn') + collect_submodules('fastapi')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('images/logos/CloudflareLogo(2560x846px).png', 'images/logos'),
        ('images/logos/LMStudioHelperCharacter(1296x912px).png', 'images/logos'),
        ('images/logos/ShimIcon(200x200px).png', 'images/logos'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ShimServer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon='images/logos/ShimIcon.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
