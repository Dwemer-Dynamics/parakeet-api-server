#!/usr/bin/env bash

set -euo pipefail

clear
cat << EOF
Parakeet STT

This will configure the Parakeet STT (Speech-to-Text) service.

Options:
* GPU / CUDA = Uses GPU acceleration for faster inference. Recommended for NVIDIA cards.
* CPU = Runs on CPU only. Use this for AMD cards or systems without GPU support.

Recommended to use GPU / CUDA if you have a Nvidia GPU.

EOF

echo "Select an option from the list:"
echo
echo "1. Enable service (GPU / CUDA)"
echo "2. Enable service (CPU)"
echo "0. Disable service"
echo

read -r -p "Select an option by picking the matching number: " selection

case "$selection" in
    0)
        echo "Disabling service. Run this again to enable it"
        rm -f /home/dwemer/parakeet-api-server/start.sh
        ;;
    1)
        ln -sf /home/dwemer/parakeet-api-server/start-gpu.sh /home/dwemer/parakeet-api-server/start.sh
        echo "[OK] Parakeet enabled with GPU / CUDA mode"
        ;;
    2)
        ln -sf /home/dwemer/parakeet-api-server/start-cpu.sh /home/dwemer/parakeet-api-server/start.sh
        echo "[OK] Parakeet enabled with CPU mode"
        ;;
    *)
        echo "Invalid selection." >&2
        exit 1
        ;;
esac
