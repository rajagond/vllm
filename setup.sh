#!/bin/bash

###########################################
# System-level updates and dependencies
###########################################

# Update package list from repositories
sudo apt-get update

# Upgrade installed packages with automatic yes to prompts
sudo apt-get upgrade -y

# Install MKL development files through apt
# Required for building packages that link against MKL
sudo apt-get install libmkl-dev -y

###########################################
# Conda environment setup
###########################################

# Install Intel Math Kernel Library (MKL) using conda
conda install mkl -y

###########################################
# Python package installation
###########################################

# Upgrade Hugging Face Hub library first 
pip install --upgrade huggingface_hub

# Install required Python packages using pip3
# - setuptools_scm: Used for managing package versions
# - prettytable: For creating formatted text tables
# - seaborn: Statistical data visualization (depends on matplotlib)
# - matplotlib: Plotting library
pip3 install setuptools_scm prettytable seaborn matplotlib

###########################################
# Project setup
###########################################

# Install current package in editable/development mode
# -vvv flag enables very verbose output for debugging
# -e . installs the package from current directory in editable mode
pip3 install -vvv -e .

# Run setup.py in development mode
# This allows to modify the code without reinstalling
python3 setup.py develop