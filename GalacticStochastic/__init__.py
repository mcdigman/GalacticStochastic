# GalacticStochastic/__init__.py
"""
A Python package for analysis and modeling of the galactic binary gravitational wave background.

GalacticStochastic
==================

This package provides tools for iterative fitting, background decomposition and noise modeling
for the Galactic Stochastic gravitational wave background, particularly in the context of LISA data analysis.

Modules
-------
global_file_index : File indexing utilities for galactic background data.
background_decomposition : Tools for decomposing the galactic background.
config_helper : Helpers for configuration management.
inclusion_state_manager : Management of which binaries are included in the different components of the background.
iterative_fit_manager : Management of iterative fit processes.
iterative_fit_state_machine : State machine for iterative fitting.
noise_manager : Noise modeling utilities.
iterative_fit : High-level interface for running or fetching iterative fit results.

Main Classes
------------
IterativeFitManager : Manages the iterative fit process and results.
BGDecomposition : Handles background decomposition for the galactic signal.

Main Function
-------------
fetch_or_run_iterative_loop : Run the entire iterative fitting procedure, including file I/O.


"""

# Optionally, set version
__version__ = '0.0.1'
