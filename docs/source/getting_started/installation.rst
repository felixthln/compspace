Installation
============

This section goes through the required steps to get started with plotting composition spaces.

Installation
------------

Install via pip
~~~~~~~~~~~~~~~~

The most convenient way of installing the package is via `PyPI <https://pypi.org/project/compspace/>`_ by simply running:

.. code-block::

   pip install compspace

Install via git
~~~~~~~~~~~~~~~~~~~

Alternatively, you can install the package from the GitHub directly. If you want to test the package in a separated
environment and you are using `Anaconda <https://www.anaconda.com>`_, we recommend to create a new conda environment
first via the following command. The package works with python 3.9 or later and was developed with 3.11.

.. code-block::

   conda create -n compspace_env python=3.11

Then clone the repository:

.. code-block::

   git clone https://github.com/felixthln/compspace.git
   cd pysimtra

...and install automatically via the local pip install command:

.. code-block::

   pip install .

Getting Started
----------------------------------------------

The package automatically registers two new projections to matplotlib: "compspace2D" and "compspace3D". To use them,
first import the package alongside matplotlib:

.. code-block::

   import matplotlib.pyplot as plt
   import compspace as cs

Then create a new figure and add a subplot with the desired projection:

.. code-block::

   fig, ax = plt.subplots(subplot_kw={'projection': 'compspace2D'})

Afterward, all matplotlib functionalities can be accessed as normal. Visit the :doc:`User Guide <../user_guide/index>`
to find some example plots.
