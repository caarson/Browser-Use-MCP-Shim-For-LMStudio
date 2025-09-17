# PyInstaller spec for LMStudio Shim GUI+Server
# Run: pyinstaller pyinstaller.spec

block_cipher = None

from PyInstaller.utils.hooks import collect_submodules
import os
import struct


def ensure_ico_from_png(png_path: str, ico_path: str):
    """Create a minimal .ico that embeds the PNG as-is (Vista+ supports PNG-in-ICO).
    If the .ico already exists, do nothing. Best-effort and safe to fail silently.
    """
    try:
        if os.path.exists(ico_path):
            return
        if not os.path.exists(png_path):
            return
        with open(png_path, 'rb') as f:
            png = f.read()
        # Verify PNG signature
        if png[:8] != b'\x89PNG\r\n\x1a\n':
            return
        # Parse IHDR to get width/height (big-endian 4-byte each)
        # Layout: 8(sig) + 4(len) + 4('IHDR') + 4(width) + 4(height)
        width = int.from_bytes(png[16:20], 'big')
        height = int.from_bytes(png[20:24], 'big')
        # ICONDIR
        header = struct.pack('<HHH', 0, 1, 1)
        # ICONDIRENTRY: width,height bytes (0 means 256). Use 32bpp defaults.
        w8 = width if 0 < width < 256 else 0
        h8 = height if 0 < height < 256 else 0
        entry = struct.pack('<BBBBHHII', w8, h8, 0, 0, 1, 32, len(png), 6 + 16)
        with open(ico_path, 'wb') as f:
            f.write(header)
            f.write(entry)
            f.write(png)
    except Exception:
        # Non-fatal; fall back to no embedded icon
        pass

# Ensure .ico exists for Windows EXE icon
ensure_ico_from_png('images/logos/ShimIcon(200x200px).png', 'images/logos/ShimIcon.ico')

hiddenimports = (
    collect_submodules('uvicorn')
    + collect_submodules('fastapi')
    + collect_submodules('psutil')
)

a = Analysis([
    'main.py',
],
    pathex=[],
    binaries=[],
    datas=[
        ('images/logos/CloudflareLogo(2560x846px).png', 'images/logos'),
        ('images/logos/LMStudioHelperCharacter(1296x912px).png', 'images/logos'),
        ('images/logos/ShimIcon(200x200px).png', 'images/logos'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ShimServer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='images/logos/ShimIcon.ico',
)
