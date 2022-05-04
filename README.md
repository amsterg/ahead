ahead
==============================

A short description of the project.

* Exploration and modelling of [AtariHead](https://zenodo.org/record/2603190) dataset for Imitation Learning & Inverse Reinforcement Learning
    
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* Containes Implementation of [Selective Gaze Augmentation](https://arxiv.org/abs/) 
    
Data Preparation
===================
* Run python [src/data/data_setup.py](ahead/src/data/data_setup.py)
    * Downloads the data into [src/data/raw](ahead/src/data/raw)
    * Creates interim data formats [src/data/interim](ahead/src/data/interim)
    * Creates Processed data [src/data/processed](ahead/src/data/processed/)
    * Does other structural setups 

Gaze prediction
===================
* CNN Model at [src/models/cnn_gaze.py](ahead/src/models/cnn_gaze.py)
    *   Run with [src/features/gaze_pred.py](ahead/src/features/gaze_pred.py)


Selective Gaze Augmented Learning
===================
* Model at [src/models/selective_gaze_only.py](ahead/src/models/selective_gaze_only.py)
    *   Run with [src/features/selective_gazed_act_pred.py](ahead/src/features/selective_gazed_act_pred.py)

Gameplay with trained model
===================
*   Run with [src/features/gazed_act_gameplay.py](ahead/src/features/gazed_act_gameplay.py)



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>