# Active Feature Acquisition - Stream

This repository holds the python code for a paper relating to active feature acquisition on streams. It is an extension of work done by Elson Serrao found under this link: https://github.com/elrasp/osm

## Structure of the Repository

additions - additional, messy code not part of framework that was used for the experiments and plots

data - the datasets that were used for the runs

osm - core code

requirements.txt - project requirements

## Datasets

The datasets used in the corresponding paper can be found on the UCI website http://archive.ics.uci.edu/ml/datasets.html

and on the MOA website https://moa.cms.waikato.ac.nz/datasets/

The cfpdss dataset was created using the additions/gen_correlated_dataset.py script

## Running the code

Run code by creating a Framework class in osm/data_streams/algorithm/framework.py and executing process_data_stream()

If unclear use additions/data_prepare.py Dataset class as guideline on how to setup a run

For further ease of use consider the use_example.bat that explains how to use the data_prepare.py code that can launch various combinations of experiment runs

## Where is what

Framework Class, entry point: osm/data_streams/algorithm/framework.py

Active Feature Acquisition methods: osm/data_streams/active_feature_acquisition/

The Supervised Merit Ranking methods are found under: osm/data_streams/active_feature_acquisition/supervised_merit_ranking/

Feature Set Selection methods: osm/data_streams/active_feature_acquistion/supervised_merit_ranking/smr_feature_set_selection.py

Active Learner: osm/data_streams/active_learner/

AFA budget managers; not used by active learners: osm/data_streams/budget_manager/

FPI imputer: osm/data_streams/imputer/PairImputer.py

FPITS method: osm/data_streams/active_feature_acquisition/supervised_merit_ranking/fpi_aided.py

Windows: osm/data_streams/windows/
