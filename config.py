from pathlib import Path

################### Data directories and paths ###################

ROOT_DIR = "/home/bw720/nde_traj"
DATA_DIR = ROOT_DIR + '/data'
OUTPUT_DIR = ROOT_DIR + '/output'
OUTPUT_BL_DIR = ROOT_DIR + '/output/baselines'
RESOURCE_DIR = ROOT_DIR + '/resources'
RPDR_2024_DIR = "/data/rpdr-ml/RPDRml_v2024.1"
PHENOTYPE_DIR = DATA_DIR + '/phenotypes'

FEAT_DIR = OUTPUT_DIR + '/features'
MODELS_DIR = OUTPUT_DIR + '/models'
RESULTS_DIR = OUTPUT_DIR + '/results'
LOGS_DIR = OUTPUT_DIR + '/logs'

FEAT_BL_DIR = OUTPUT_BL_DIR + '/features'
MODELS_BL_DIR = OUTPUT_BL_DIR + '/models'
RESULTS_BL_DIR = OUTPUT_BL_DIR + '/results'
LOGS_BL_DIR = OUTPUT_BL_DIR + '/logs'

base_data_dict = {
    "DX_orig_file": RPDR_2024_DIR + "/RPDRml__DX.txt",
    "MED_orig_file": RPDR_2024_DIR + "/RPDRml__MED.txt",
    "LAB_orig_file": RPDR_2024_DIR + "/RPDRml__LAB.txt",
    "PROC_orig_file": RPDR_2024_DIR + "/RPDRml__PROC.txt",
    "DEMO_orig_file": RPDR_2024_DIR + "/RPDRml__demographics.txt",

    "MGBB_pids_file": RESOURCE_DIR + "/mgbb_pids.txt",
    "DX_dict_file": RPDR_2024_DIR + "/RPDRml__DX_dictionary.txt",
    "MED_dict_file": RPDR_2024_DIR + "/RPDRml__MED_dictionary.txt",
    "LAB_dict_file": RPDR_2024_DIR + "/RPDRml__LAB_dictionary.txt",
    "PROC_dict_file": RPDR_2024_DIR + "/RPDRml__PROC_dictionary.txt",

    "DX_base_file": DATA_DIR + "/base_data/RPDRml__DX_base_v2024.1.pkl",
    "MED_base_file": DATA_DIR + "/base_data/RPDRml__MED_base_v2024.1.pkl",
    "LAB_base_file": DATA_DIR + "/base_data/RPDRml__LAB_base_v2024.1.pkl",
    "PROC_base_file": DATA_DIR + "/base_data/RPDRml__PROC_base_v2024.1.pkl",
    # "ALL_base_file": DATA_DIR + "/base_data/RPDRml__ALL_base_v2024.1.pkl",
}


################### Parameters and settings ###################

random_state = 42

feat_params = {
    "feat_lvl": "visit",
    "feat_scaling": False,
    "min_df": 100,
    "max_df": 0.8,
    "feat_type": "demo_dx_med",
    "feat_transform": "bow",
    "label_name": "label",
    "custom_bins":{
        'age': [0, 25, 40, 60, 75, float('inf')],  # age groups
    }
}