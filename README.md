# README #

### What is this repository for? ###

This repository contains a submission for Scientific Computing EMAT30008: using finite difference methods to
solve partial differential equations.

### How do I get set up? ###

See the report "report.pdf" for a description of the software, along with the Jupyter notebooks
"Examples.ipynb" and "Solutions.ipynb" that go with it.

The bulk of the code belongs in 

parabolicpde.py - parabolic equation solvers and objects
hyperbolicpde.py - hyperbolic equation solvers and objects
ellipticpde.py - elliptic equation solvers and objects

Extra functions can be found in 

boundary.py - boundary condition objects
visualizations.py - tools for plotting/animating
helpers.py - useful functions that don't live anywhere else

The files worksheets.py and examples.py were used in the process of verifying the solvers and answering
the worksheet questions. Most of this code ended up in the two notebooks.

James Cass
jc4196@bristol.ac.uk