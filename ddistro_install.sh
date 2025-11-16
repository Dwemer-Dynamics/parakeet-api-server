#!/bin/bash

cd /home/dwemer/parakeet-api-server
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

./install.sh


