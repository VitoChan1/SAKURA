Installation
============
SAKURA is developed using python~3.9 and PyTorch 1.13.1.

.. note::
    To avoid potential dependency conflicts, installing within a conda environment is recommended.

.. _environment:

Environment setting
-------------------
We assume conda is installed. Run the following command to create a new environment named `sakura`:

.. code-block:: console

    conda create -n sakura python=3.9

Activate the environment:

.. code-block:: console

    conda activate sakura

.. _installation:

Install SAKURA via cloning
-------------------------------
Clone the repository to your local directory:

.. code-block:: console

    git clone https://github.com/Yip-Lab/SAKURA.git
    cd SAKURA

Then, install the `sakura` package via pip:

.. code-block:: console

    (sakura) user:~/.../SAKURA/$
    pip install .

The project dependencies (managed by Poetry) are already included and wrapped by Projen; installation may take a few moments.

**Now you are all set.** Proceed to learn how to use the sakura package.
