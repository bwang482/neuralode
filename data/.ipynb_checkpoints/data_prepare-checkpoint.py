import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from scipy.sparse import vstack as csr_vstack
from sklearn.model_selection import train_test_split

import common
from code_mapping import *
from feature_compute import compute_features
from src.config import base_data_dict, data_dict, random_state, feat_params
from src.utils.general_functs import save_pickle, load_pickle, combine_lists, calc_date_diff_v2


t0 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cohort', help='Cohort selection', default='MDD')
args = parser.parse_args()


def combine_data(df_dx, df_demo, join_how="outer"):
    df = df_dx
    phe_code_lst = []
    phe_date_lst = []
    for i, row in tqdm(df.iterrows()):
        dx_dates, dx_codes = common.deduplicate_order_w_nan(row['sstart_date'], row['concept_codes'])
        # lab_dates, lab_codes = common.deduplicate_order_w_nan(row['sstart_date_lab'], row['concept_codes_h_lab'])
        # med_dates, med_codes = common.deduplicate_order_w_nan(row['sstart_date_med'], row['concept_codes_h_med'])
        # proc_dates, proc_codes = common.deduplicate_order_w_nan(row['sstart_date_proc'], row['concept_codes_h_proc'])
        assert(len(dx_dates) == len(dx_codes))
        # assert(len(lab_dates) == len(lab_codes))
        # assert(len(med_dates) == len(med_codes))
        # assert(len(proc_dates) == len(proc_codes))

        phe_codes, phe_dates = phe_mapping(dx_codes, dx_dates)
        # rxnorm_codes, rxnorm_dates = rxnorm_mapping(med_codes, med_dates)
        # cpt4_codes, cpt4_dates = cpt4_mapping(proc_codes, proc_dates)
        # loinc_codes, loinc_dates = loinc_mapping(lab_codes, lab_dates)
        assert(len(phe_dates) == len(phe_codes))
        # assert(len(rxnorm_dates) == len(rxnorm_codes))
        # assert(len(cpt4_dates) == len(cpt4_codes))
        # assert(len(loinc_dates) == len(loinc_codes))
        phe_code_lst.append(phe_codes)
        phe_date_lst.append(phe_dates)
        # rxnorm_code_lst.append(rxnorm_codes)
        # rxnorm_date_lst.append(rxnorm_dates)
        # cpt4_code_lst.append(cpt4_codes)
        # cpt4_date_lst.append(cpt4_dates)
        # loinc_code_lst.append(loinc_codes)
        # loinc_date_lst.append(loinc_dates)

        # all_dates = phe_dates + loinc_dates + rxnorm_dates + cpt4_dates
        # all_codes = phe_codes + loinc_codes + rxnorm_codes + cpt4_codes

        # combined = combine_lists(all_dates, all_codes)
        # combined_dates.append([x[0] for x in combined])
        # combined_codes.append([[xi for xl in x[1] for xi in xl] for x in combined])
    df['dx_codes'] = phe_code_lst
    df['dx_dates'] = phe_date_lst
    # df['concept_id_h_rxnorm'] = rxnorm_code_lst
    # df['start_date_rxnorm'] = rxnorm_date_lst
    # df['concept_id_h_cpt4'] = cpt4_code_lst
    # df['start_date_cpt4'] = cpt4_date_lst
    # df['concept_id_h_loinc'] = loinc_code_lst
    # df['start_date_loinc'] = loinc_date_lst
    # df['concept_id_h'] = combined_codes
    # df['start_date'] = combined_dates

    df_combined = pd.merge(df, 
                        df_demo[['subject_num', 'gender', 'sbirth_date', 'race', 
                                 'veteran', 'marital_status', 'ethnicity']], 
                        left_on='subject_num', right_on='subject_num')
    df_combined['sbirth_date'] = pd.to_datetime(df_combined.sbirth_date)
    df_combined['age'] = df_combined.apply(lambda x: calc_date_diff_v2(datetime.now(), x['sbirth_date']), axis=1)
    df_combined = df_combined.drop(['sbirth_date'], axis=1)
    return df_combined


def create_base_data(base_data_dict):
    DX_file = base_data_dict["DX_orig_file"]
    # MED_file = base_data_dict["MED_orig_file"]
    # LAB_file = base_data_dict["LAB_orig_file"]
    # PROC_file = base_data_dict["PROC_orig_file"]
    DEMO_file = base_data_dict["DEMO_orig_file"]
    CASE_file = base_data_dict["CASE_file"]

    logger.info("Loading original data files")
    df_dx = pd.read_csv(DX_file, sep='\t')
    # df_med = pd.read_csv(MED_file, sep='\t')
    # df_lab = pd.read_csv(LAB_file, sep='\t')
    # df_proc = pd.read_csv(PROC_file, sep='\t')
    df_demo = pd.read_csv(DEMO_file, sep='\t')

    df_dx.drop(['diagnosis_type', 'site', 'epic_fhir_ws'], axis=1, inplace=True)
    dx_pids = df_dx.subject_num.unique().tolist()
    df_dx = df_dx[df_dx['subject_num'].isin(dx_pids[:2000])] # Randomly sample 2000 patients

    df_case = pd.read_pickle(CASE_file)
    case_pids = df_case[df_case['label']==True].subject_num.unique().tolist()
    print(len(case_pids), "cases from the current definition")

    logger.info("Grouping datasets by PID")
    df_dx = common.groupby_pid(df_dx, case_pids, "DX")
    # df_med = common.groupby_pid(df_med, case_pids, "MED")
    # df_lab = common.groupby_pid(df_lab, case_pids, "LAB")
    # df_proc = common.groupby_pid(df_proc, case_pids, "PROC")

    logger.info("Grouping concepts by dates")
    df_dx = common.group_concepts(df_dx, new_col='concept_codes')
    # df_med = common.group_concepts(df_med)
    # df_lab = common.group_concepts(df_lab)
    # df_proc = common.group_concepts(df_proc)

    logger.info("Saving dataframes to .pkl")
    save_pickle(df_dx, base_data_dict["DX_base_file"])
    # save_pickle(df_med, base_data_dict["MED_base_file"])
    # save_pickle(df_lab, base_data_dict["LAB_base_file"])
    # save_pickle(df_proc, base_data_dict["PROC_base_file"])

    logger.info("Combining data modalities")
    DF = combine_data(df_dx, df_demo, "outer")
    print("Number of controls =", DF['label'].value_counts()[0])
    print("Number of cases =", DF['label'].value_counts()[1])

    logger.info("Saving combined dataframe to .pkl")
    save_pickle(DF, base_data_dict["ALL_base_file"])


def data_split(datapath, data_dict, label_col="label"):
    print("Loading data")
    df = load_pickle(datapath)

    print("Adding new pids")
    df['pid'] = df.index

    print("Spliting train and test data")
    df_train, df_test, _, _ = train_test_split(
        df, 
        df[label_col], 
        test_size=feat_params['test_size'], 
        stratify=df[label_col],
        random_state=random_state
    )
    save_pickle(df_train.reset_index(drop=True), data_dict['train_data_file'])
    save_pickle(df_test.reset_index(drop=True), data_dict['test_data_file'])


def feature_extraction(data_dict, feat_params):

    trainpath = data_dict['train_data_file']
    testpath = data_dict['test_data_file']
    train_featpath = data_dict['train_feat_file']
    test_featpath = data_dict['test_feat_file']
    featname_path = data_dict['featname_file']

    print("Loading train & test data frames")
    df_train = load_pickle(trainpath)
    df_test = load_pickle(testpath)
    
    print("Computing features")
    X_train_cov, X_test_cov, cov_names, X_train_obs, X_test_obs, obs_names = compute_features(
        df_train, df_test, 
        feat_type=feat_params["feat_type"], 
        feat_transform=feat_params["feat_transform"],
        feat_lvl=feat_params["feat_lvl"], 
        max_df=feat_params["max_df"], 
        min_df=feat_params["min_df"]
    )
    feature_names = {
        "cov": cov_names,
        "obs": obs_names
    }

    print("Data formatting")
    time_since = []
    patient_ids = []
    for i, row in df_train.iterrows():
        assert( len(row['dx_dates'])==X_train_obs[i].shape[0] )
        time_since.append( [(d-row['dx_dates'][0]).days for d in row['dx_dates']] )
        patient_ids.append( [row['pid'] for d in row['dx_dates']] )

    train_dict = {
        "cov": X_train_cov.toarray(),
        "obs": csr_vstack(X_train_obs),
        "time": np.array([dt for pt in time_since for dt in pt]),
        "patient_id": np.array([pids for pt in patient_ids for pids in pt]),
        "labels": np.array([1 if df_train['label'][i]==1 else 0 for i in range(len(patient_ids)) for pids in patient_ids[i]])
    }

    time_since = []
    patient_ids = []
    for i, row in df_test.iterrows():
        assert( len(row['dx_dates'])==X_test_obs[i].shape[0] )
        time_since.append( [(d-row['dx_dates'][0]).days for d in row['dx_dates']] )
        patient_ids.append( [row['pid'] for d in row['dx_dates']] )

    test_dict = {
        "cov": X_test_cov.toarray(),
        "obs": csr_vstack(X_test_obs),
        "time": np.array([dt for pt in time_since for dt in pt]),
        "patient_id": np.array([pids for pt in patient_ids for pids in pt]),
        # "labels": np.array(df_test['label'].tolist())
        "labels": np.array([1 if df_test['label'][i]==1 else 0 for i in range(len(patient_ids)) for pids in patient_ids[i]])
    }

    print(f"Cov feature size = {test_dict['cov'].shape[1]} dimensions")
    print(f"Obs feature size = {test_dict['obs'].shape[1]} dimensions")

    print("Saving train & test features to .pkl")
    save_pickle(train_dict, train_featpath)
    save_pickle(test_dict, test_featpath)
    save_pickle(feature_names, featname_path)


def main():
    # logger.info("Create base data")
    # create_base_data(base_data_dict)
    # logger.info("Data split")
    # data_split(base_data_dict['ALL_base_file'], data_dict)

    logger.info("Feature extraction")
    feature_extraction(data_dict, feat_params)

    print(f"Overall took {(time.time()-t0)/60} mins.") 


if __name__ == "__main__":
    main()