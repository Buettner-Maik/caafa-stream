@echo off
rem -d to specify data sets
rem -i to specify iterations [rangeStart..rangeEnd] e.g. '0..1' for iteration 0
rem -m to specify missingness' [missingness]+
rem -p to breate new batches from the raw data overwriting the old iteration [[True, False], batch size, ild size]
rem -c to specify the cost distribution [equal, increasing, decreasing, cheaper_cats, cheaper_nums, ...]+
rem -t to specify tasks [lower, upper, [RA, SWAED, MWAED, SWIG, SWSU, ...]+[SBM, IPF, NBM, ...]+[KBSMRFSS, KPBSMRFSS, KQGSMRFSS, KRSMRFSS, ...], ...]+
rem -s to specify the maximum set size of the acquistion set the feature set selection method can choose
rem -b to specify budget in groups of five [threshold, budget gained once, budget gained per batch, budget gained per instance, budget gained per successful acquisition]+
rem threshold takes any float or one out of the defined calculated thresholds: [imax, half, mean]
rem -w to specify window [size in batches]
rem --overwrite to automatically overwrite old results
rem --noconfirm to skip the confirmation step

REM static data sets
rem -p True 50 50

REM stream data sets
rem -p False 50 50


REM example runs of static data sets
start python -W ignore additions\data_prepare.py -d adult magic nursery pendigits -i 0 1 -m 0.25 0.5 0.75 -p True 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d adult magic nursery pendigits -i 2 3 -m 0.25 0.5 0.75 -p True 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d adult magic nursery pendigits -i 4 5 -m 0.25 0.5 0.75 -p True 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d adult magic nursery pendigits -i 6 7 -m 0.25 0.5 0.75 -p True 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d adult magic nursery pendigits -i 8 9 -m 0.25 0.5 0.75 -p True 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10

REM example runs of stream data sets
start python -W ignore additions\data_prepare.py -d cfpdss electricity -i 0 1 -m 0.25 0.5 0.75 -p False 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d cfpdss electricity -i 2 3 -m 0.25 0.5 0.75 -p False 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d cfpdss electricity -i 4 5 -m 0.25 0.5 0.75 -p False 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d cfpdss electricity -i 6 7 -m 0.25 0.5 0.75 -p False 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d cfpdss electricity -i 8 9 -m 0.25 0.5 0.75 -p False 50 50 -c equal increasing decreasing -t lower upper SWAED_FPITS_0.3+IPF+KBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+SBM+KQGSMRFSS -s 1 2 4 100 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10



cmd /k