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
start python -W ignore additions\data_prepare.py -d abalone adult magic nursery occupancy pendigits -i 0..5 -m 0.125 0.25 0.375 0.5 0.625 0.75 0.875 -p True 50 50 -c equal increasing -t lower upper RA+SBM+KBSMRFSS SWAED+IPF+KPBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+IPF+KQGSMRFSS -s 1 2 4 8 16 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d abalone adult magic nursery occupancy pendigits -i 5..10 -m 0.125 0.25 0.375 0.5 0.625 0.75 0.875 -p True 50 50 -c equal increasing -t lower upper RA+SBM+KBSMRFSS SWAED+IPF+KPBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+IPF+KQGSMRFSS -s 1 2 4 8 16 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d electricity sea -i 0..10 -m 0.125 0.25 0.375 0.5 0.625 0.75 0.875 -p False 50 50 -c equal increasing -t lower upper RA+SBM+KBSMRFSS SWAED+IPF+KPBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+IPF+KQGSMRFSS -s 1 2 4 8 16 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d evenoddf3d3 evenoddf5d3 evenoddf7d3 evenoddf9d3 -i 0..10 -m 0.125 0.25 0.375 0.5 0.625 0.75 0.875 -p True 50 50 -c equal increasing -t lower upper RA+SBM+KBSMRFSS SWAED+IPF+KPBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+IPF+KQGSMRFSS -s 1 2 4 8 16 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
start python -W ignore additions\data_prepare.py -d evenoddf9d2 evenoddf9d4 evenoddf9d5 -i 0..10 -m 0.125 0.25 0.375 0.5 0.625 0.75 0.875 -p True 50 50 -c equal increasing -t lower upper RA+SBM+KBSMRFSS SWAED+IPF+KPBSMRFSS SWIG+TCIPF+KRSMRFSS SWSU+IPF+KQGSMRFSS -s 1 2 4 8 16 -b mean 0 0 0.5 0 mean 0 0 1 0 mean 0 0 2 0 -w 10
cmd /k