# PyInstaller spec for LMStudio Shim GUI+Server
# Run: pyinstaller pyinstaller.spec

block_cipher = None

from PyInstaller.utils.hooks import collect_submodules

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
)
