#!/bin/bash

# Update package list from repositories
sudo apt-get update

# Upgrade installed packages with automatic yes to prompts
sudo apt-get upgrade -y

# Install required Python packages using pip3
# - setuptools_scm: Used for managing package versions
# - prettytable: For creating formatted text tables
# - seaborn: Statistical data visualization
# - matplotlib: Plotting library
pip3 install setuptools_scm prettytable seaborn matplotlib

# Install current package in editable/development mode
# -vvv flag enables very verbose output for debugging
# -e . installs the package from current directory in editable mode
pip3 install -vvv -e .

# Run setup.py in development mode
python3 setup.py develop