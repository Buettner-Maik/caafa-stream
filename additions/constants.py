import osm.data_streams.constants as osmc

# plotting consts
AFAS = ['no_AFA', 'RA', 'MWAED', 'SWAED', 'SWIG', 'SWSU', 'SWAED_T', 'SWAED_Q', 'SWAED_I', 'SWAED_D', 'SWAED_C', 'SWAED_P', 'SWAED_E', 'SWAED_H', 'SWAED_H2', 'SWIG_H2', 'SWSU_H2', 'SWAED_IMAX', 'SWAED_IEXP']
NAFA = AFAS[0]
RA = AFAS[1]
MWAED = AFAS[2]
SWAED = AFAS[3]
SWIG = AFAS[4]
SWSU = AFAS[5]
SWAED_T = AFAS[6]
SWAED_Q = AFAS[7]
SWAED_I = AFAS[8]
SWAED_D = AFAS[9]
SWAED_C = AFAS[10]
SWAED_P = AFAS[11]
SWAED_E = AFAS[12]
SWAED_H = AFAS[13]
SWAED_H2 = AFAS[14]
SWIG_H2 = AFAS[15]
SWSU_H2 = AFAS[16]
SWAED_IMAX = AFAS[17]
SWAED_IEXP = AFAS[18]

FSS = ['KRSMRFSS', 'KBSMRFSS', 'KPBSMRFSS', 'KQSMRFSS', 'KQGSMRFSS', 'KBAQGSMRFSS']
KRSMRFSS = FSS[0]
KBSMRFSS = FSS[1]
KPBSMRFSS = FSS[2]
KQSMRFSS = FSS[3]
KQGSMRFSS = FSS[4]
KBAQGSMRFSS = FSS[5]

AFA_FSS = AFAS[:2] + [afa + '+' + fss for afa in AFAS[2:] for fss in FSS]

BMS = ['NBM', 'SBM', 'IPF', 'TCIPF']
NBM = BMS[0]
SBM = BMS[1]
IPF = BMS[2]
TCIPF = BMS[3]

MCS = ['0.125', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875']
COSTDISTS = ['equal', 'increasing', 'decreasing', 'cheaper_cats', 'cheaper_nums']
SET_SIZES = [1, 2, 3, 4, 6, 16]
BUDGETS = ['i1.0', 'i2.0', 'i3.0', 'i4.0', 'i5.0']
DATASETS = ['abalone', 'adult', 'electricity', 'forest', 'intrusion', 'magic', 'nursery', 'occupancy', 'pendigits', 'sea']
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
DIR_PLOTS = "data/plots/auto"

# PLOT CONSTS
REF_DF = "results.pkl.gzip"
INCOMPL_DF = "incompl.pkl.gzip"

#SUMMARY STATS
STAT_ACQUISITION_TIME = 'acq_time'
STAT_SAMPLE_TIME = 'sample_time'
STAT_TEST_TIME = 'test_time'
STAT_TRAIN_TIME = 'train_time'
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
STATS = [STAT_ACQUISITION_TIME, STAT_SAMPLE_TIME, STAT_TEST_TIME, STAT_TRAIN_TIME, STAT_QUERIED, STAT_ANSWERED, STAT_BUDGET, STAT_BUDGET_SPENT, STAT_BUDGET_USED, STAT_ACCURACY, STAT_PRECISION, STAT_RECALL, STAT_F1, STAT_LOG_LOSS, STAT_KAPPA]
STAT_FEATURE_MISSING = 'missing '
STAT_FEATURE_QUERIED = 'queried '
STAT_FEATURE_ANSWERED = 'answered '
STAT_FEATURE_MERIT = 'merit '

STIN_ACQUISITION_TIME = (osmc.time_stats, osmc.acquisition_time)
STIN_SAMPLE_TIME = (osmc.time_stats, osmc.sample_time)
STIN_TEST_TIME = (osmc.time_stats, osmc.test_time)
STIN_TRAIN_TIME = (osmc.time_stats, osmc.train_time)
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
