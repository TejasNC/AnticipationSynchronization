# Simulating Kuramoto Model

This repository contains code for simulating the Kuramoto model, a mathematical model used to describe synchronization phenomena in systems of coupled oscillators.

## Organization of the Code

### Files 

- `kuramoto.py`: Contains the class implementation of the Kuramoto model
- `animations.py`: Run it to view animations for different values of $K$ and $R$
- `r_vs_k.ipynb`: Contains plot of $R$ against $K$ to find $K_C$
- `simulating_kuramoto.ipynb`: Constains results of simlations on different values of $K$ and $R$
- 

## Mathematical Concept

The Kuramoto model is a set of coupled differential equations used to describe the synchronization of a large set of oscillators. Each oscillator has its own natural frequency, and they are coupled through a sine function of their phase differences. The model is given by:

$$
    \frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_i) 
$$

where:
- $\theta_i$ is the phase of the \(i\)-th oscillator,
- $\omega_i$ is the natural frequency of the \(i\)-th oscillator,
- $K$ is the coupling constant,
- $N$ is the total number of oscillators.

The Kuramoto model demonstrates how synchronization can emerge in a system of oscillators with different natural frequencies due to coupling.

## How to Run the Simulation

1. Clone the repository:
    ```bash
    git clone https://github.com/TejasNC/.git
    cd Simulating_Kuramoto
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run this script for viewing the animations:
    ```bash
    python animations.py
    ```

4. Results of some simulations are discussed in [simulating_kuramoto.ipynb](simulating_kuramoto.ipynb)

5. Plot of order parameter $R$ vs $K$ can be viewd in [r_vs_k.ipynb](r_vs_k.ipynb). 

## References

- Kuramoto_model. (n.d.). https://en.wikipedia.org/wiki/Kuramoto_model.
- Strogatz, S. H. (2000). From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators. Physica D: Nonlinear Phenomena, 143(1-4), 1-20. https://doi.org/10.1016/S0167-2789(00)00094-4

Feel free to explore the code and modify the parameters to see how they affect the synchronization behavior of the oscillators.
