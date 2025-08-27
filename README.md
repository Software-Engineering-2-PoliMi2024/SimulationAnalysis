# Simulation Analysis

This directory contains scripts and tools for analyzing simulation data.

## simulation loader

the `simulation_loader` module provides functionality to load and preprocess simulation data from various sources.
In particular, it includes:

- `retrive_simulations` : a function to retrive simulation inputs and outpus packed together as a pandas DataFrame.
- `retrieve_simulations_io` : a function to retrive simulation inputs and outputs as separate pandas DataFrames.

an example of how to use the `MongoLoader` to load simulation data from a MongoDB database is provided in the `Example` directory.
