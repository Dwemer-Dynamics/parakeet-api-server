#!/bin/bash
clear
cat << EOF
parakeet

TO-DO: Explain here

EOF

echo "0. Disable service";
echo "1. Enable service (GPU)"
echo "2. Enable service (CPU)"

# Prompt the user to make a selection
read -p "Select an option by picking the matching number: " selection

# Validate the input

if [ "$selection" -eq "0" ]; then
    echo "Disabling service. Run this again to enable"
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





