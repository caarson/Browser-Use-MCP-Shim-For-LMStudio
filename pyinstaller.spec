# PyInstaller spec for LMStudio Shim GUI+Server
# Run: pyinstaller pyinstaller.spec

block_cipher = None

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('uvicorn') + collect_submodules('fastapi')

a = Analysis([
    'main.py',
],
    pathex=[],
    binaries=[],
    datas=[],
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
    [],
    exclude_binaries=True,
    name='lmstudio_shim',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='lmstudio_shim',
)
