# Real-time Optimization

This repository provides examples on how to use the `rtotools` package to build and evaluate different kinds of RTO systems.

## Installation
Installing the rtotools package should be sufficient for running the examples:

```bash
pip install packages/rtotools-0.1.0-py3-none-any.whl
```

Note: not available in pypi yet!

## CCTA 2021
See the following notebook that was used to create the results displayed in the paper: `notebooks/MA_GaussianProcesses_CCTA_2021.ipynb`

If you you want to reproduce the results, create two databases using the script above and then run the file `scripts/magp_experiment.py`. Results might differ a bit due to the stochastic nature of the system.

Don't forget to adjust the file names in both the notebook and the script file. Don't hesitate to send me a message if you have any troubles.

## References

Darby, M. L., Nikolaou, M., Jones, J., & Nicholson, D. (2011). RTO: An overview and assessment of current practice. *Journal of Process Control*, 21(6), 874-884.

A. Marchetti, B. Chachuat, and D. Bonvin (2009)  Modifier-adaptation methodology for real-time optimization *Industrial \& engineering chemistry research* vol. 48, no. 13, pp. 6022–6033, 2009

de Avila Ferreira, T., Shukla, H. A., Faulwasser, T., Jones, C. N., & Bonvin, D. (2018, June). Real-time optimization of uncertain process systems via modifier adaptation and Gaussian processes. In *2018 European Control Conference (ECC)* (pp. 465-470). IEEE.