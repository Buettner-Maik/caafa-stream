import osm.data_streams.constants as osmc

# plotting consts
# feature importance methods
AFAS = []
AFAS.append('no_AFA'); NAFA = AFAS[-1]
AFAS.append('RA'); RA = AFAS[-1]
AFAS.append('PCFI'); PCFI = AFAS[-1]

# supervised merit ranking
AFAS.append('MWAED'); MWAED = AFAS[-1]
AFAS.append('SWAED'); SWAED = AFAS[-1]
AFAS.append('SWIG'); SWIG = AFAS[-1]
AFAS.append('SWSU'); SWSU = AFAS[-1]
AFAS.append('PCFI_DTC'); PCFI_DTC = AFAS[-1]

# budget threshold tests
AFAS.append('SWAED_NDB'); SWAED_NDB = AFAS[-1]
AFAS.append('SWAED_IMAX'); SWAED_IMAX = AFAS[-1]
AFAS.append('SWAED_IEXP'); SWAED_IEXP = AFAS[-1]

# Quality altered
AFAS.append('SWAED_SSBQ'); SWAED_SSBQ = AFAS[-1] # SetSizeBiased Coefficient multiplied
AFAS.append('SWAED_SSBQ_FPI'); SWAED_SSBQ_FPI = AFAS[-1]
AFAS.append('SWAED_IMPQ'); SWAED_IMPQ = AFAS[-1] # SumMerits * Sum(1-NRMSES) * SetSizeBias
AFAS.append('SWAED_IMPQ2'); SWAED_IMPQ2 = AFAS[-1] # SumMerits * Sum(1-NRMSES) * SetSizeCostBias

# feature pair imputer based or aided
AFAS.append('NAFA_FPI'); NAFA_FPI = AFAS[-1]
AFAS.append('RA_FPI'); RA_FPI = AFAS[-1]
AFAS.append('SWAED_FPI'); SWAED_FPI = AFAS[-1]

AFAS.append('SWAED_FPITS'); SWAED_FPITS = AFAS[-1]
AFAS.append('SWAED_FPITS_0.1'); SWAED_FPITS_10 = AFAS[-1]
AFAS.append('SWAED_FPITS_0.2'); SWAED_FPITS_20 = AFAS[-1]
AFAS.append('SWAED_FPITS_0.3'); SWAED_FPITS_30 = AFAS[-1]
AFAS.append('SWAED_FPITS_0.4'); SWAED_FPITS_40 = AFAS[-1]
AFAS.append('SWAED_FPITS_0.5'); SWAED_FPITS_50 = AFAS[-1]
AFAS.append('SWAED_FPITS_0.6'); SWAED_FPITS_60 = AFAS[-1]
AFAS.append('SWAED_FPITS_0.7'); SWAED_FPITS_70 = AFAS[-1]
AFAS.append('SWAED_FPITS_0.8'); SWAED_FPITS_80 = AFAS[-1]
AFAS.append('SWAED_FPITS_0.9'); SWAED_FPITS_90 = AFAS[-1]

AFAS.append('SWAED_FPITS_LOG_0.1'); SWAED_FPITS_LOG_10 = AFAS[-1]
AFAS.append('SWAED_FPITS_LOG_0.2'); SWAED_FPITS_LOG_20 = AFAS[-1]
AFAS.append('SWAED_FPITS_LOG_0.3'); SWAED_FPITS_LOG_30 = AFAS[-1]
AFAS.append('SWAED_FPITS_LOG_0.4'); SWAED_FPITS_LOG_40 = AFAS[-1]
AFAS.append('SWAED_FPITS_LOG_0.5'); SWAED_FPITS_LOG_50 = AFAS[-1]
AFAS.append('SWAED_FPITS_LOG_0.6'); SWAED_FPITS_LOG_60 = AFAS[-1]
AFAS.append('SWAED_FPITS_LOG_0.7'); SWAED_FPITS_LOG_70 = AFAS[-1]
AFAS.append('SWAED_FPITS_LOG_0.8'); SWAED_FPITS_LOG_80 = AFAS[-1]
AFAS.append('SWAED_FPITS_LOG_0.9'); SWAED_FPITS_LOG_90 = AFAS[-1]


# AFAS.append('SWAED_IMPTS'); SWAED_IMPTS = AFAS[-1]
# AFAS.append('SWAED_IMPTS_2'); SWAED_IMPTS_2 = AFAS[-1]
# AFAS.append('SWAED_IMPTS_6'); SWAED_IMPTS_6 = AFAS[-1]
# AFAS.append('SWAED_IMPTS_10'); SWAED_IMPTS_10 = AFAS[-1]

# AFAS = ['no_AFA', 'RA', 'MWAED', 'SWAED', 'SWIG', 'SWSU', 'SWAED_T', 'SWAED_Q', 'SWAED_I', 'SWAED_D', 'SWAED_C', 'SWAED_P', 'SWAED_E', 'SWAED_H', 'SWAED_H2', 'SWIG_H2', 'SWSU_H2', 'SWAED_IMAX', 'SWAED_IEXP', 'FIPC', 'SWAEDFCC', 'SWAED_AQ', 'SWAED_II', 'SWAEDFCC_II', 'SWAED_RPRI', 'SWAEDFCC_RPRI', 'SWAED_SPRI', 'SWAEDFCC_SPRI', 'SWAED_TRPRI', 'SWAED_TRPRI_0', 'SWAED_TRPRI_25', 'SWAED_TRPRI_50', 'SWAED_TRPRI_75', 'SWAED_TRPRI_100', 'SWAED_TRPRI_200', 'SWAED_BPRI', 'SWAED_IQ', 'SWAED_QT', 'SWAED_FPI', 'RA_FPI', 'no_AFA_FPI']
# NAFA = AFAS[0]
# RA = AFAS[1]
# MWAED = AFAS[2]
# SWAED = AFAS[3]
# SWIG = AFAS[4]
# SWSU = AFAS[5]
# SWAED_T = AFAS[6]
# SWAED_Q = AFAS[7]
# SWAED_I = AFAS[8]
# SWAED_D = AFAS[9]
# SWAED_C = AFAS[10]
# SWAED_P = AFAS[11]
# SWAED_E = AFAS[12]
# SWAED_H = AFAS[13]
# SWAED_H2 = AFAS[14]
# SWIG_H2 = AFAS[15]
# SWSU_H2 = AFAS[16]
# SWAED_IMAX = AFAS[17]
# SWAED_IEXP = AFAS[18]
# FIPC = AFAS[19]
# SWAEDFCC = AFAS[20]
# SWAED_AQ = AFAS[21]
# SWAED_II = AFAS[22]
# SWAEDFCC_II = AFAS[23]
# SWAED_RPRI = AFAS[24]
# SWAEDFCC_RPRI = AFAS[25]
# SWAED_SPRI = AFAS[26]
# SWAEDFCC_SPRI = AFAS[27]
# SWAED_TRPRI = AFAS[28]
# SWAED_TRPRI_0 = AFAS[29]
# SWAED_TRPRI_25 = AFAS[30]
# SWAED_TRPRI_50 = AFAS[31]
# SWAED_TRPRI_75 = AFAS[32]
# SWAED_TRPRI_100 = AFAS[33]
# SWAED_TRPRI_200 = AFAS[34]
# SWAED_BPRI = AFAS[35]
# SWAED_IQ = AFAS[36]
# SWAED_QT = AFAS[37]
# SWAED_FPI = AFAS[38]
# RA_FPI = AFAS[39]
# NAFA_FPI = AFAS[40]

# SMR feature set selectors
FSS = []
FSS.append('KRSMRFSS'); KRSMRFSS = FSS[-1]
FSS.append('KBSMRFSS'); KBSMRFSS = FSS[-1]
FSS.append('KPBSMRFSS'); KPBSMRFSS = FSS[-1]
FSS.append('KQSMRFSS'); KQSMRFSS = FSS[-1] #Maybe remove?
FSS.append('KQGSMRFSS'); KQGSMRFSS = FSS[-1]
FSS.append('KBAQGSMRFSS'); KBAQGSMRFSS = FSS[-1] #NotImplemented
FSS.append('KBIAMSMRFSS'); KBIAMSMRFSS = FSS[-1] #Make Merit SWAED method
FSS.append('KBITSMRFSS'); KBITSMRFSS = FSS[-1]

# FSS = ['KRSMRFSS', 'KBSMRFSS', 'KPBSMRFSS', 'KQSMRFSS', 'KQGSMRFSS', 'KBAQGSMRFSS', 'KBIAMSMRFSS', 'KBITSMRFSS']
# KRSMRFSS = FSS[0]
# KBSMRFSS = FSS[1]
# KPBSMRFSS = FSS[2]
# KQSMRFSS = FSS[3]
# KQGSMRFSS = FSS[4]
# KBAQGSMRFSS = FSS[5]
# KBIAMSMRFSS = FSS[6]
# KBITSMRFSS = FSS[7]

AFA_FSS = AFAS[:2] + [afa + '+' + fss for afa in AFAS[2:] for fss in FSS]

BMS = []
BMS.append('NBM'); NBM = BMS[-1]
BMS.append('SBM'); SBM = BMS[-1]
BMS.append('IPF'); IPF = BMS[-1]
BMS.append('TCIPF'); TCIPF = BMS[-1]

# BMS = ['NBM', 'SBM', 'IPF', 'TCIPF']
# NBM = BMS[0]
# SBM = BMS[1]
# IPF = BMS[2]
# TCIPF = BMS[3]

MCS = ['0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875']
COSTDISTS = ['equal', 'increasing', 'decreasing', 'cheaper_cats', 'cheaper_nums']
SET_SIZES = [1, 2, 3, 4, 6, 16]
BUDGETS = ['i1.0', 'i2.0', 'i3.0', 'i4.0', 'i5.0']
DATASETS = ['abalone', 'adult', 'cfpdss', 'electricity', 'forest', 'intrusion', 'magic', 'nursery', 'occupancy', 'pendigits', 'sea']
CLASSIFIERS = ['sgd', 'dtc']
KEEP_NORMS = [True, False]
WINDOW_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

REF_DF_FILTERS = ['dataset', 'i', 'm', 'cdist', 'afa', 'fss', 'ss', 'bm', 'b', 'al', 'flag']
COLEXTFUNCS = ['mean', 'max', 'min', 'last', 'keep']
# extensions
EXT_ARFF = ".arff"

# file paths
DIR_ARFF = "data/arff"
DIR_CSV = "data/csv"
DIR_GEN = "data/gen"
DIR_RPG = "data/arff/rpg"
DIR_PLOTS = "plots"

# PLOT CONSTS
REF_DF = "results.pkl.gzip"
INCOMPL_DF = "incompl.pkl.gzip"

#SUMMARY STATS
STAT_ACQUISITION_TIME = 'acq_time'
STAT_SAMPLE_TIME = 'sample_time'
STAT_TEST_TIME = 'test_time'
STAT_TRAIN_TIME = 'train_time'
STAT_ACQUISITIONS = 'acqs'
STAT_BUDGET_SAVED = 'b_saved'
STAT_QUERIED = 'queried'
STAT_ANSWERED = 'answered'
STAT_BUDGET = 'b'
STAT_BUDGET_SPENT = 'b_spent'
STAT_BUDGET_USED = 'b_used'
STAT_ACCURACY = 'acc'
STAT_PRECISION = 'precision'
STAT_RECALL = 'recall'
STAT_F1 = 'f1'
STAT_LOG_LOSS = 'log_loss'
STAT_KAPPA = 'kappa'
STATS = [STAT_ACQUISITION_TIME, STAT_SAMPLE_TIME, STAT_TEST_TIME, STAT_TRAIN_TIME, STAT_ACQUISITIONS, STAT_BUDGET_SAVED, STAT_QUERIED, STAT_ANSWERED, STAT_BUDGET, STAT_BUDGET_SPENT, STAT_BUDGET_USED, STAT_ACCURACY, STAT_PRECISION, STAT_RECALL, STAT_F1, STAT_LOG_LOSS, STAT_KAPPA]
STAT_FEATURE_MISSING = 'missing '
STAT_FEATURE_QUERIED = 'queried '
STAT_FEATURE_ANSWERED = 'answered '
STAT_FEATURE_MERIT = 'merit '

STIN_ACQUISITION_TIME = (osmc.time_stats, osmc.acquisition_time)
STIN_SAMPLE_TIME = (osmc.time_stats, osmc.sample_time)
STIN_TEST_TIME = (osmc.time_stats, osmc.test_time)
STIN_TRAIN_TIME = (osmc.time_stats, osmc.train_time)
STIN_ACQUISITIONS = (osmc.feature_pair_imputer_stats, osmc.acquisitions_skipped)
STIN_BUDGET_SAVED = (osmc.feature_pair_imputer_stats, osmc.budget_saved)
STIN_QUERIED = (osmc.budget_manager_stats, osmc.queried)
STIN_ANSWERED = (osmc.budget_manager_stats, osmc.answered)
STIN_BUDGET = (osmc.budget_manager_stats, osmc.budget)
STIN_BUDGET_SPENT = (osmc.budget_manager_stats, osmc.budget_spent)
STIN_BUDGET_USED = (osmc.budget_manager_stats, osmc.budget_used)
STIN_ACCURACY = (osmc.summary_level, osmc.accuracy)
STIN_PRECISION = (osmc.summary_level, osmc.precision)
STIN_RECALL = (osmc.summary_level, osmc.recall)
STIN_F1 = (osmc.summary_level, osmc.f1)
STIN_LOG_LOSS = (osmc.summary_level, osmc.log_loss)
STIN_KAPPA = (osmc.summary_level, osmc.kappa)
STIN_FEATURE_MISSING = osmc.missing_features
STIN_FEATURE_QUERIED = osmc.active_feature_acquisition_queries
STIN_FEATURE_ANSWERED = osmc.active_feature_acquisition_answers
STIN_FEATURE_MERIT = osmc.active_feature_acquisition_merits
