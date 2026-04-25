#!/bin/bash

# Dynamically scans all 'lib' directories inside NVIDIA packages
# installed in the virtual environment and concatenates them separated by ':'
NVIDIA_LIBS=$(find "$(pwd)/.venv/lib/python3.13/site-packages/nvidia" \
    -maxdepth 2 -type d -name 'lib' | tr '\n' ':')

# Ensures the .vscode folder exists
mkdir -p .vscode

# Generates settings.json with the correct paths for this machine.
# NVIDIA paths are prefixed to Flatpak's original /app/lib,
# ensuring the linker finds them before anything else.
cat > .vscode/settings.json << JSON
{
    "terminal.integrated.env.linux": {
        "LD_LIBRARY_PATH": "${NVIDIA_LIBS}/app/lib"
    }
}
JSON

echo "✅ .vscode/settings.json generated successfully!"
echo "   Open a new terminal in VS Code for the settings to take effect."
