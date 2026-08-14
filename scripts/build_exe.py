"""
PyInstaller Build Script to package the Stock Screening System into an Executable (.exe).
Run: python scripts/build_exe.py
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def build_executable():
    print("=" * 60)
    print("Building Executable (.exe) for Stock Screening System...")
    print("=" * 60)

    app_main = BASE_DIR / "src" / "dashboard" / "app.py"
    output_dir = BASE_DIR / "dist"
    style_css = BASE_DIR / "src" / "dashboard" / "style.css"
    models_dir = BASE_DIR / "models"

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=StockScreeningSystem",
        "--onefile",
        "--clean",
        f"--add-data={style_css};src/dashboard",
        f"--add-data={models_dir};models",
        str(app_main),
    ]

    print("Running command:", " ".join(pyinstaller_cmd))
    try:
        subprocess.run(pyinstaller_cmd, cwd=str(BASE_DIR), check=True)
        print("\n" + "=" * 60)
        print("Build Successful! Executable created in 'dist/StockScreeningSystem.exe'")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"Build Failed: {e}")


if __name__ == "__main__":
    build_executable()
