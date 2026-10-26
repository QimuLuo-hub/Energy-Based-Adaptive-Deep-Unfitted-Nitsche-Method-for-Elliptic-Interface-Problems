# Energy-Based Adaptive Deep Unfitted Nitsche Method for Elliptic Interface Problems

**Published research** · *East Asian Journal on Applied Mathematics* (2026)

[Read the published article](https://doi.org/10.4208/eajam.2025-069.100925)

This repository accompanies research on solving **elliptic partial differential equations with interfaces** using a neural approximation, a Nitsche-based treatment of interface conditions, and adaptive training-point selection.

## Why interface problems?

Interface problems arise when material properties or governing coefficients change across an internal boundary. The solution may remain continuous while its derivatives or fluxes behave differently on either side. Accurately representing these conditions is important for numerical solutions, particularly with irregular interfaces and strongly contrasting coefficients.

## Method overview

The proposed **energy-based adaptive deep unfitted Nitsche method** combines three ideas:

1. **Neural solution approximation.** A neural representation approximates the solution in the subdomains separated by an interface.
2. **Unfitted Nitsche formulation.** Interface conditions are incorporated through the variational/energy formulation, avoiding a requirement that the computational representation conform exactly to the interface.
3. **Energy-based adaptive sampling.** Training points are redistributed toward regions identified as more demanding by the energy-based criterion, rather than relying solely on uniform sampling.

Together, these components target accurate interface resolution with more effective use of training samples.

## Numerical investigation

The paper evaluates the method on **two- and three-dimensional elliptic interface problems**, considering interface geometry, coefficient contrasts, and comparisons with non-adaptive sampling. The reported experiments show improvements in accuracy and sampling efficiency for the tested problems. For the exact model settings, error definitions, figures, and comparisons, refer to the published article.

## How to use this repository

This repository contains the research materials associated with the method. To understand the mathematical formulation and experimental setup, **start with the paper**. When navigating the code, look for the components that define the PDE and interface, implement the energy/Nitsche loss, select training samples, and evaluate numerical errors.

The original research implementation may depend on experiment-specific parameters or file paths. No universal installation or execution command is claimed here; use the settings documented in the source files and paper for individual experiments.

## Publication

**Energy-Based Adaptive Deep Unfitted Nitsche Method for Elliptic Interface Problems.** *East Asian Journal on Applied Mathematics* (2026). [DOI: 10.4208/eajam.2025-069.100925](https://doi.org/10.4208/eajam.2025-069.100925)
