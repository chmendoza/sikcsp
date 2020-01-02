import os
import sys
# Add path to package directory to access main module using absolute import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import si_k_means
import datasets
