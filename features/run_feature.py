import sys
sys.path.append("/home/bw720/nde_traj/src")
import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from config import feat_params
from config import DATA_DIR, FEAT_DIR
from utils.general_functs import save_pickle, load_pickle
from features.feature_utils import combine_df
from features.feature_compute import compute_features


t0 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cohort', type=str, help='Cohort selection', default='MDD')
parser.add_argument('-lw', '--lookback_window', type=int, help='Lookback window', default=1)
args = parser.parse_args()


def create_data_dict(df, X, pid_col="subject_num"):
    if feat_params["feat_type"] == "demo_dx_med_proc_lab":
        date_col = "all_dates"
    elif feat_params["feat_type"] == "demo_dx":
        date_col = "dx_dates"
    elif feat_params["feat_type"] == "demo_dx_med":
        date_col = "dx_med_dates"

    time_since = []
    patient_ids = []
    labels = []
    for i, row in df.iterrows():
        assert( len(row[date_col])==X[i].shape[0] )
        time_since.append( [(d-row[date_col][0]).days for d in row[date_col]] )
        patient_ids.append(row[pid_col])
        labels.append(row['label'])
        
    assert(len(time_since) == len(patient_ids))
    assert(len(patient_ids) == len(X))

    data = [
        (
            str(patient_ids[i]),
            time_since[i],  
            X[i], 
            labels[i]
            
        ) for i in range(len(patient_ids))
    ]
    return data
    

def print_summary(df):
    df_cas = df[df['label'] == 1]
    df_ctl = df[df['label'] == 0]
    print(f"Average age for cases = {np.mean(df_cas['age'])}")
    print(f"Average age for controls = {np.mean(df_ctl['age'])}")
    num_days_cas = df_cas['all_dates'].apply(
        lambda x: (x[-1] - x[0]).days if x else None
    ).dropna().tolist()
    num_days_ctl = df_ctl['all_dates'].apply(
        lambda x: (x[-1] - x[0]).days if x else None
    ).dropna().tolist()
    print(f"Average length of EHR history (in days) for cases = {np.mean(num_days_cas)}")
    print(f"Average length of EHR history (in days) for controls = {np.mean(num_days_ctl)}")
    num_tps_cas = df_cas['all_dates'].apply(
        lambda x: len(x) if x else None
    ).dropna().tolist()
    num_tps_ctl = df_ctl['all_dates'].apply(
        lambda x: len(x) if x else None
    ).dropna().tolist()
    print(f"Average number of time points (visits) for cases = {np.mean(num_tps_cas)}")
    print(f"Average number of time points (visits) for controls = {np.mean(num_tps_ctl)}\n")
    print()
    dx_dates = df['all_dates'].explode()
    print(f"Earliest visit date = {dx_dates.min()}")
    print(f"Latest visit date = {dx_dates.max()}")


def feature_extraction(feat_params):
    DATA_path = os.path.join(DATA_DIR, args.cohort.upper())
    FEAT_path = os.path.join(FEAT_DIR, args.cohort.upper())

    train_data_filename = f"RPDRml__{args.cohort.upper()}_train.{args.lookback_window}y.pk"
    val_data_filename = f"RPDRml__{args.cohort.upper()}_val.{args.lookback_window}y.pk"
    test_data_filename = f"RPDRml__{args.cohort.upper()}_test.{args.lookback_window}y.pk"
    train_feat_filename = f"train.{feat_params['feat_type']}.{args.lookback_window}y.pk"
    val_feat_filename = f"val.{feat_params['feat_type']}.{args.lookback_window}y.pk"
    test_feat_filename = f"test.{feat_params['feat_type']}.{args.lookback_window}y.pk"
    train_val_feat_filename = f"train_val.{feat_params['feat_type']}.{args.lookback_window}y.pk"
    featname_filename = f"feature_names.{feat_params['feat_type']}.{args.lookback_window}y.pk"

    trainpath = os.path.join(DATA_path, train_data_filename)
    valpath = os.path.join(DATA_path, val_data_filename)
    testpath = os.path.join(DATA_path, test_data_filename)
    train_featpath = os.path.join(FEAT_path, train_feat_filename)
    val_featpath = os.path.join(FEAT_path, val_feat_filename)
    test_featpath = os.path.join(FEAT_path, test_feat_filename)
    train_val_featpath = os.path.join(FEAT_path, train_val_feat_filename)
    featname_path = os.path.join(FEAT_path, featname_filename)

    logger.info("Loading training, validation & test data frames")
    print("Loading from", trainpath)
    df_train = load_pickle(trainpath)
    print("Loading from", valpath)
    df_val = load_pickle(valpath)
    print("Loading from", testpath)
    df_test = load_pickle(testpath)
    print("Training data size:", len(df_train))
    print(df_train.label.value_counts())
    print("Validation data size:", len(df_val))
    print(df_val.label.value_counts())
    print("Test data size:", len(df_test))
    print(df_test.label.value_counts())
    print()

    logger.info("Combining codes and dates")
    df_train = combine_df(df_train)
    df_val = combine_df(df_val)
    df_test = combine_df(df_test)

    print_summary(df_train) # Print summary stats for training data
    
    logger.info("Computing features")
    X_train, X_test, feat_names = compute_features(
        df_train, df_test, 
        feat_type=feat_params["feat_type"], 
        feat_transform=feat_params["feat_transform"],
        feat_lvl="visit", # visit-level fix-sized feature representation
        feat_scaling=feat_params["feat_scaling"],
        max_df=feat_params["max_df"], 
        min_df=feat_params["min_df"]
    )

    X_train, X_val, _ = compute_features(
        df_train, df_val, 
        feat_type=feat_params["feat_type"], 
        feat_transform=feat_params["feat_transform"],
        feat_lvl=feat_params["feat_lvl"], # visit-level fix-sized feature representation
        feat_scaling=feat_params["feat_scaling"],
        max_df=feat_params["max_df"], 
        min_df=feat_params["min_df"]
    )

    print(X_train[0].toarray().max())
    print(X_val[-1].toarray().max())
    print(X_train[0].toarray().min())
    print(X_val[-1].toarray().min())
    print("\n")

    logger.info("Generating data dicts")
    train_data = create_data_dict(df_train, X_train)
    val_data = create_data_dict(df_val, X_val)
    test_data = create_data_dict(df_test, X_test)
    print(len(train_data), "training samples")
    print(len(val_data), "validation samples")
    print(len(test_data), "test samples\n")

    logger.info("Saving training, validation & test features to pickle files")
    print(train_featpath)
    print(val_featpath)
    print(test_featpath)
    print(train_val_featpath)
    print(featname_path)

    save_pickle(train_data, train_featpath)
    save_pickle(val_data, val_featpath)
    save_pickle(test_data, test_featpath)
    save_pickle(feat_names, featname_path)
    print("\nAll saved")


def main():
    t0 = time.time()
        
    logger.info(f"Feature extraction for {args.cohort.upper()} cohort")
    feature_extraction(
        feat_params,
    )

    print(f"Overall took {(time.time()-t0)/60} mins.") 


if __name__ == '__main__':
    main()