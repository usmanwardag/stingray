X-Ray Timing Made Easy
=======================

We are writing a general-purpose timing package for X-ray time series.

Authors
--------
* Abigail Stevens (UvA)
* Daniela Huppenkothen (NYU CDS)

Contents
--------

Currently implemented:
- make a light curve from event data
- make periodograms in Leahy and rms normalization
- average periodograms
- maximum likelihood fitting of periodograms/parametric models

To be added soon:
- cross spectra and lags (time vs energy, time vs frequency)
- bispectra (?)
- cross correlation functions, coherence
- spectral-timing functionality
- Bayesian QPO searches
- power colours
- rms spectra

Prerequisites
-------------

Stingray is designed to be able to run with a minimum 
of dependencies, but certain functionality depends on 
additional packages.

Required prerequisites are:
- numpy (version?)
- scipy (version?)

Additional packages that can be helpful for certain 
functions, but are not strictly required:

- testing: nosetests (HIGHLY RECOMMENDED) --> (LINK)
- documentation: sphinx (HIGHLY RECOMMENDED) --> (LINK)
- plotting: matplotlib (add link)
- corner plots: corner.py (add link)
- sampling: emcee (add link)


Documentation
-------------

Is generated using `Sphinx`_. Try::

   $ sphinx-build doc doc/_build

Then open ``./doc/_build/index.html`` in the browser of your choice.

.. _Sphinx: http://sphinx-doc.org

Test suite
----------

Try::

   $ nosetests

Copyright
---------

All content © 2015 the authors. The code is distributed under the MIT license.

Pull requests are welcome! If you are interested in the further development of
this project, please `get in touch via the issues
<https://github.com/dhuppenkothen/stingray/issues>`_!
