# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A handwritten digit recognizer, developed as two separate versions:

- `desktop_version/`: a Tkinter app with a NumPy MLP trained on MNIST. It owns training and the model file.
- `web_version/`: a static HTML/JS page that runs the desktop version's trained model in the browser (weights exported by `web_version/export_weights.py`).

Each folder has its own `CLAUDE.md` with version-specific commands and architecture. Read it before working in that folder.

## Rules for all versions

- Write all code and comments in English.
- Every newly created file must start with a comment giving its creation date and time in local time (KST, UTC+9), using that file type's comment syntax, e.g. `# Created: 2026-10-18 16:54 (KST)` in Python or `<!-- Created: ... -->` in Markdown/HTML. Get the time with `Get-Date -Format "yyyy-MM-dd HH:mm"` instead of guessing.
