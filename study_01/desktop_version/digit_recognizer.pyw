#! python3.12
# The shebang above tells the Windows Python launcher (pyw.exe) to use
# Python 3.12, which has NumPy and Pillow installed.
"""
Double-click launcher for the handwritten digit recognizer.

Files with the .pyw extension run with pythonw.exe, so no console window
appears. Any error is shown in a message box because there is no console
to print it to.
"""

import os
import sys
import traceback

# Make sure digit_recognizer.py next to this file can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from digit_recognizer import main

    sys.argv = sys.argv[:1]  # Ignore arguments passed by Explorer
    main()
except Exception:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Digit Recognizer - Error", traceback.format_exc())
    root.destroy()
