#!/bin/bash
clear
cat << EOF
Parakeet STT

This will configure the Parakeet TTS (Text-to-Speech) service..

Options:
* GPU = Uses GPU acceleration for faster inference. Recommended for NVIDIA cards.
* CPU = Runs on CPU only. Use this for AMD cards or systems without GPU support.

Recommended to use GPU if you have a Nvidia GPU.

EOF

echo "Select an option from the list:"
echo
echo "1. Enable service (GPU)"
echo "2. Enable service (CPU)"
echo "0. Disable service"
echo

# Prompt the user to make a selection
read -p "Select an option by picking the matching number: " selection

# Validate the input

if [ "$selection" -eq "0" ]; then
    echo "Disabling service. Run this again to enable it"
    rm /home/dwemer/parakeet-api-server/start.sh 
    exit 1
fi

if [ "$selection" -eq "1" ]; then
    ln -sf /home/dwemer/parakeet-api-server/start-gpu.sh /home/dwemer/parakeet-api-server/start.sh
    exit 1
fi

if [ "$selection" -eq "2" ]; then
    ln -sf /home/dwemer/parakeet-api-server/start-cpu.sh /home/dwemer/parakeet-api-server/start.sh
    exit 1
fi





