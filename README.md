ahead
==============================

A short description of the project.

* Exploration and modelling of [AtariHead](https://zenodo.org/record/2603190) dataset for Inverse Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Data Preparation
===================
* Run python [src/data/data_setup.py](ahead/src/data/data_setup.py)
    * Downloads the data into [src/data/raw](ahead/src/data/raw)
    * Creates interim data formats [src/data/interim](ahead/src/data/interim)
    * Creates Processed data [src/data/processed](ahead/src/data/processed/) - TODO ( The models below can still be run without this.)
    * Does other structural setups 

Gaze prediction
===================
* CNN Model at [src/models/cnn_gaze.py](ahead/src/models/cnn_gaze.py)
    *   Run with [src/features/gaze_pred.py](ahead/src/features/gaze_pred.py)
* MDN model at [src/models/mdn.py](ahead/src/models/mdn.py) - In process

Supervised action learner
===================
* Model at [src/models/action_sl.py](ahead/src/models/action_sl.py)
    *   Run with [src/features/act_pred.py](ahead/src/features/act_pred.py)

Reinforcement Learning
===================
 * TODO


Inverse Reinforcement Learning
===================
* TODO



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>