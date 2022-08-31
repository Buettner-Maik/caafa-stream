# Active Feature Acquisition - Stream

This repository holds the python code for a paper relating to active feature acquisition on streams. It is an extension of work done by Elson Serrao found under this link: https://github.com/elrasp/osm
and an iteration on the framework found under: https://github.com/Buettner-Maik/afa-stream

## Structure of the Repository

additions - additional code not part of framework that was used for the experiments and plots

data - raw data of the data sets for the runs; to run experiments preprocessed batches have to be generated first
Alternatively copy the prepared data sets from https://github.com/Buettner-Maik/afa-stream/tree/master/data/csv

osm - core code

requirements.txt - project requirements

## Datasets

The datasets used in the corresponding paper can be found on the UCI website http://archive.ics.uci.edu/ml/datasets.html

The evenodd data sets have been generated using a python script found under data/csv/evenodd

## Running the code

Run code by creating a Framework class in osm/data_streams/algorithm/framework.py and executing process_data_stream()

The framework requires a pandas.DataFrame summary file as a pointer to an initial batch of data and each successively processed batch within the stream

The additions/data_prepare.py file is an example on how to setup a run

You find an example of how to use the data_prepare.py file in the use_example.bat in the root of this repository

## Where is what

Framework Class, entry point: osm/data_streams/algorithm/framework.py

Active Feature Acquisition: osm/data_streams/active_feature_acquisition/

Active Learner: osm/data_streams/active_learner/

AFA budget managers; not used by active learners: osm/data_streams/budget_manager/

Feature Set Selection methods for supervised_merit_ranking: osm/data_streams/active_feature_acquistion/supervised_merit_ranking/smr_feature_set_selection.py

Windows: osm/data_streams/windows/
