# Simulating Lorentz System

This project simulates the Lorentz system, a set of differential equations used to model atmospheric convection. The Lorentz system is a simplified mathematical model for atmospheric convection, introduced by Edward Lorenz in 1963. It is notable for having chaotic solutions for certain parameter values and initial conditions.

## Code Files

### `lorentz_system.py`

This file contains the implementation of the Lorentz system equations. It defines the differential equations and provides functions to solve them using numerical methods.

### `sensitive_dependence.py`

This file has the visualiation of the sensitive dependence on initial conditions property of the chaotic Lorentz System. The results are available as `sensitive_dependence_time_series.png` and `sensitive_dependece.gif` in the [Results Folder](../Results/). 

## Results

The `Results` folder contains the output of the code files. It includes plots of the time series and phase space.

## Usage

To run the code, execute the following commands in the [Code Folder](./Code):

```bash
python lorentz_system.py
```

This will generate the plots of the Lorentz system solutions.

```bash
python sensitive_dependece.py 
```

This will generate the plots showing sensitive dependence on initial conditions of chaotic Lorentz system.

## Dependencies

This project requires the following Python libraries:

- `numpy`
- `matplotlib`
- `scipy`

You can install these libraries using `pip`:

```bash
pip install requirements.txt
```
