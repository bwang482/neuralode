from pathlib import Path

################### Data directories and paths ###################

ROOT_DIR = "/home/bw720/nde_traj"
DATA_DIR = ROOT_DIR + '/data'
OUTPUT_DIR = ROOT_DIR + '/output'
OUTPUT_BL_DIR = ROOT_DIR + '/output/baselines'
RESOURCE_DIR = ROOT_DIR + '/resources'
RPDR_2024_DIR = "/data/rpdr-ml/RPDRml_v2024.1"

FEAT_DIR = OUTPUT_DIR + '/features'
MODELS_DIR = OUTPUT_DIR + '/models'
RESULTS_DIR = OUTPUT_DIR + '/results'
LOGS_DIR = OUTPUT_DIR + '/logs'

FEAT_BL_MDD_DIR = OUTPUT_BL_DIR + '/mdd/features'
MODELS_BL_MDD_DIR = OUTPUT_BL_DIR + '/mdd/models'
RESULTS_BL_MDD_DIR = OUTPUT_BL_DIR + '/mdd/results'
LOGS_BL_MDD_DIR = OUTPUT_BL_DIR + '/mdd/logs'

FEAT_BL_CAD_DIR = OUTPUT_BL_DIR + '/cad/features'
MODELS_BL_CAD_DIR = OUTPUT_BL_DIR + '/cad/models'
RESULTS_BL_CAD_DIR = OUTPUT_BL_DIR + '/cad/results'
LOGS_BL_CAD_DIR = OUTPUT_BL_DIR + '/cad/logs'

base_data_dict = {
    "DX_orig_file": RPDR_2024_DIR + "/RPDRml__DX.txt",
    "NLP_orig_file": RPDR_2024_DIR + "/RPDRml__NLPSymptoms.txt",
    "MED_orig_file": RPDR_2024_DIR + "/RPDRml__MED.txt",
    "LAB_orig_file": RPDR_2024_DIR + "/RPDRml__LAB.txt",
    "PROC_orig_file": RPDR_2024_DIR + "/RPDRml__PROC.txt",
    "DEMO_orig_file": RPDR_2024_DIR + "/RPDRml__demographics.txt",

    "DX_dict_file": RPDR_2024_DIR + "/RPDRml__DX_dictionary.txt",
    "MED_dict_file": RPDR_2024_DIR + "/RPDRml__MED_dictionary.txt",
    "LAB_dict_file": RPDR_2024_DIR + "/RPDRml__LAB_dictionary.txt",
    "PROC_dict_file": RPDR_2024_DIR + "/RPDRml__PROC_dictionary.txt",

    "DX_base_file": DATA_DIR + "/base_data/RPDRml__DX_base_v2024.1.pkl",
    "MED_base_file": DATA_DIR + "/base_data/RPDRml__MED_base_v2024.1.pkl",
    "LAB_base_file": DATA_DIR + "/base_data/RPDRml__LAB_base_v2024.1.pkl",
    "PROC_base_file": DATA_DIR + "/base_data/RPDRml__PROC_base_v2024.1.pkl",
    "ALL_base_file": DATA_DIR + "/base_data/RPDRml__ALL_base_v2024.1.pkl",

    "MDD_file": RESOURCE_DIR + "/MDD_pids.pk", #MDD pids and index dates
    "CAD_file": RESOURCE_DIR + "/CAD_pids.pk", #CAD pids and index dates
}

data_dict_mdd = {
    "data_file": DATA_DIR + "/RPDRml__MDD_v2024.1.pkl", 
    "train_data_file": DATA_DIR + "/RPDRml__MDD_train.pk",
    "test_data_file": DATA_DIR + "/RPDRml__MDD_test.pk",
    "train_feat_file": DATA_DIR + "/data_dict_train.pk",
    "test_feat_file": DATA_DIR + "/data_dict_test.pk",
    "featname_file": DATA_DIR + "/feature_names.pk",
}

data_dict_cad = {
    "data_file": DATA_DIR + "/RPDRml__CAD_v2024.1.pkl", 
    "train_data_file": DATA_DIR + "/RPDRml__CAD_train_xsm.pk",
    "test_data_file": DATA_DIR + "/RPDRml__CAD_test_xsm.pk",
    "train_feat_file": DATA_DIR + "/data_dict_train.pk",
    "test_feat_file": DATA_DIR + "/data_dict_test.pk",
    "featname_file": DATA_DIR + "/feature_names.pk",
}

data_dict_baseline_mdd = {
    "data_file": DATA_DIR + "/RPDRml__MDD_v2024.1.pkl", 
    "train_data_file": DATA_DIR + "/RPDRml__MDD_train.pk",
    "test_data_file": DATA_DIR + "/RPDRml__MDD_test.pk",
    "train_feat_file": FEAT_BL_MDD_DIR + "/train_feats.pk",
    "test_feat_file": FEAT_BL_MDD_DIR + "/test_feats.pk",
    "train_y_file": FEAT_BL_MDD_DIR + "/train_y.pkl",
    "test_y_file": FEAT_BL_MDD_DIR + "/test_y.pkl",
    "featname_file": FEAT_BL_MDD_DIR + "/feature_names.pk",
    "label_name": "label"
}

data_dict_baseline_cad = {
    "data_file": DATA_DIR + "/RPDRml__CAD_v2024.1.pkl", 
    "train_data_file": DATA_DIR + "/RPDRml__CAD_train.pk",
    "test_data_file": DATA_DIR + "/RPDRml__CAD_test.pk",
    "train_feat_file": FEAT_BL_CAD_DIR + "/train_feats.pk",
    "test_feat_file": FEAT_BL_CAD_DIR + "/test_feats.pk",
    "train_y_file": FEAT_BL_CAD_DIR + "/train_y.pkl",
    "test_y_file": FEAT_BL_CAD_DIR + "/test_y.pkl",
    "featname_file": FEAT_BL_CAD_DIR + "/feature_names.pk",
    "label_name": "label"
}

################### Parameters and settings ###################

random_state = 42

feat_params = {
    "test_size": 0.2,
    "feat_lvl": "visit",
    "feat_scaling": False,
    "min_df": 20,
    "max_df": 0.8,
    "feat_type": "demo_dx_med_proc_lab",
    "feat_transform": "bow",
}