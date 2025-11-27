import sys
sys.path.append("../")
import os
import time
import argparse
import numpy as np
import pandas as pd
from loguru import logger

import common
from code_mapping import *
from config import base_data_dict, feat_params, DATA_DIR, PHENOTYPE_DIR, random_state
from utils.general_functs import save_txt, save_pickle, load_pickle, combine_lists


t0 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cohort', type=str, help='Cohort selection', default='MDD')
parser.add_argument('-cd', '--cutoff_date', type=str, help='Data cutoff date', default="1900-01-01")
parser.add_argument('-lw', '--lookback_window', type=int, help='Lookback window', default=1)
parser.add_argument('-mr', '--matching_ratio', type=int, help='Matching ratio', default=1)
parser.add_argument('-df', '--data_folder', type=str, help='Data folder name', default="random_visit_w_filters_all_matched")
args = parser.parse_args()


def create_base_data(base_data_dict):
    """ Create base dataframe """
    DX_file = base_data_dict["DX_orig_file"]
    MED_file = base_data_dict["MED_orig_file"]
    LAB_file = base_data_dict["LAB_orig_file"]
    PROC_file = base_data_dict["PROC_orig_file"]
    DEMO_file = base_data_dict["DEMO_orig_file"]
    BASE_data_path = os.path.join(DATA_DIR, args.cohort.upper(), 'base_data')

    logger.info("Loading original data files")
    df_dx = pd.read_csv(DX_file, sep='\t', 
                        usecols=['subject_num', 'concept_code', 'sstart_date'])
    df_med = pd.read_csv(MED_file, sep='\t', 
                         usecols=['subject_num', 'concept_code', 'sstart_date'])
    df_lab = pd.read_csv(LAB_file, sep='\t', 
                         usecols=['subject_num', 'concept_code', 'sstart_date', 'valueflag', 'valtype', 'nval'])
    df_proc = pd.read_csv(PROC_file, sep='\t', 
                          usecols=['subject_num', 'concept_code', 'sstart_date'])
    df_demo = pd.read_csv(DEMO_file, sep='\t')

    print(f"Loading {args.cohort} case and control data")
    df_cohort = pd.read_csv(os.path.join(PHENOTYPE_DIR, f"{args.cohort}_pids.csv"), dtype={2: "str"})

    print(f"Filtering data by data cutoff date")
    # We only want cases whose index events occurred on and after cutoff_date
    df_cohort, cohort_pids = common.filter_cohort(df_cohort, cutoff_date=args.cutoff_date) # filter based on index date criteria

    logger.info(f"Selecting only {args.cohort} patients")
    df_dx = df_dx[df_dx['subject_num'].isin(cohort_pids)]
    df_med = df_med[df_med['subject_num'].isin(cohort_pids)]
    df_lab = df_lab[df_lab['subject_num'].isin(cohort_pids)]
    df_proc = df_proc[df_proc['subject_num'].isin(cohort_pids)]
    df_demo = df_demo[df_demo['subject_num'].isin(cohort_pids)]

    logger.info("Mapping codes")
    dx_dict = pd.read_csv(base_data_dict['DX_dict_file'], sep="\t", dtype={'PheCode': str})
    med_dict = pd.read_csv(base_data_dict['MED_dict_file'], sep="\t")
    lab_dict = pd.read_csv(base_data_dict['LAB_dict_file'], sep="\t")
    proc_dict = pd.read_csv(base_data_dict['PROC_dict_file'], sep="\t")

    df_dx = phecode_mapping(df_dx, dx_dict)
    df_med = rxnorm_mapping(df_med, med_dict)
    df_lab = loinc_mapping(df_lab, lab_dict)
    df_proc = cpt4_mapping(df_proc)

    logger.info("Grouping datasets by PID")
    df_dx = common.groupby_pid(df_dx, df_cohort, "DX", args.cutoff_date) # apply cut-off date to DX
    df_med = common.groupby_pid(df_med, df_cohort, "MED", args.cutoff_date) # apply cut-off date to MED
    df_lab = common.groupby_pid(df_lab, df_cohort, "LAB", args.cutoff_date) # apply cut-off date to LAB
    df_proc = common.groupby_pid(df_proc, df_cohort, "PROC", args.cutoff_date) # apply cut-off date to PROC

    logger.info("Grouping concepts by dates")
    df_dx = common.group_concepts(df_dx)
    df_med = common.group_concepts(df_med)
    df_lab = common.group_concepts(df_lab)
    df_proc = common.group_concepts(df_proc)

    logger.info("Balancing case and control numbers")
    mgbb_pids = np.loadtxt(base_data_dict['MGBB_pids_file'])
    df_dx, df_med, df_lab, df_proc = common.balance_case_control_data(
        df_dx, df_med, df_lab, df_proc, mgbb_pids, random_state=random_state
    )

    cohort_pids = df_dx.subject_num.unique().tolist()
    output_file = f"{args.cohort.upper()}_balanced_pids.txt"
    output_path = os.path.join(BASE_data_path, output_file)
    save_txt(cohort_pids, output_path)
    print(f"Saved balanced patient IDs to {output_path}\n")

    # logger.info("Saving dataframes of different data types in pickle format")
    # save_pickle(df_dx, os.path.join(BASE_data_path, "RPDRml__DX_base_v2024.1.pkl"))
    # save_pickle(df_med, os.path.join(BASE_data_path, "RPDRml__MED_base_v2024.1.pkl"))
    # save_pickle(df_lab, os.path.join(BASE_data_path, "RPDRml__LAB_base_v2024.1.pkl"))
    # save_pickle(df_proc, os.path.join(BASE_data_path, "RPDRml__PROC_base_v2024.1.pkl"))
    # print("Saved")

    logger.info("Combining data types")
    df_final = combine_data(df_dx, df_med, df_lab, df_proc, df_demo, "outer")
    print("Number of controls =", df_final['label'].value_counts()[0])
    print("Number of cases =", df_final['label'].value_counts()[1])

    logger.info("Saving combined dataframe in pickle format")
    save_pickle(df_final, os.path.join(BASE_data_path, "RPDRml__ALL_base_v2024.2.pkl"))
    print("Saved\n\n")


def combine_data(df_dx, df_med, df_lab, df_proc, df_demo, join_how="outer"):
    """ Combine multiple dataframes """
    df1 = pd.merge(
        df_dx.drop(['concept_codes'], axis=1),
        df_med.drop(['concept_codes'], axis=1), 
        on='subject_num', suffixes=('_dx', '_med'), how=join_how
    )
    df1['index_date'] = df1['index_date_dx'].fillna(df1['index_date_med'])
    df1['label'] = df1['label_dx'].fillna(df1['label_med'])

    df2 = pd.merge(
        df_lab.drop(['concept_codes', 'label', 'index_date'], axis=1),
        df_proc.drop(['concept_codes', 'label', 'index_date'], axis=1), 
        on='subject_num', suffixes=('_lab', '_proc'), how=join_how
    )

    df = pd.merge(df1, df2, on='subject_num', how=join_how)

    def process_row(row):
        # Extract and deduplicate each type
        phe_dates, phe_codes = common.deduplicate_dates_with_codes(
            row['start_dates_dx'], row['concepts_by_date_dx'])
        rxnorm_dates, rxnorm_codes = common.deduplicate_dates_with_codes(
            row['start_dates_med'], row['concepts_by_date_med'])
        cpt4_dates, cpt4_codes = common.deduplicate_dates_with_codes(
            row['start_dates_proc'], row['concepts_by_date_proc'])
        loinc_dates, loinc_codes = common.deduplicate_dates_with_codes(
            row['start_dates_lab'], row['concepts_by_date_lab'])
        
        assert(len(phe_codes) == len(phe_dates))
        assert(len(rxnorm_codes) == len(rxnorm_dates))
        assert(len(cpt4_codes) == len(cpt4_dates))
        assert(len(loinc_codes) == len(loinc_dates))
        
        # Combine all dates and codes
        all_dates = phe_dates + loinc_dates + rxnorm_dates + cpt4_dates
        all_codes = phe_codes + loinc_codes + rxnorm_codes + cpt4_codes
        combined = combine_lists(all_dates, all_codes)
        
        return pd.Series({
            'dx_codes': phe_codes,
            'dx_dates': phe_dates,
            'med_codes': rxnorm_codes,
            'med_dates': rxnorm_dates,
            'proc_codes': cpt4_codes,
            'proc_dates': cpt4_dates,
            'lab_codes': loinc_codes,
            'lab_dates': loinc_dates,
            'all_dates': [x[0] for x in combined],
            'all_codes': [[xi for xl in x[1] for xi in xl] for x in combined]
        })
    
    df_temp = df.apply(process_row, axis=1)
    df = pd.concat([df[['subject_num', 'index_date', 'label']], 
                    df_temp], axis=1)
    
    # Combine with demographic data
    demo_cols = [
        'subject_num', 'gender', 'sbirth_date', 'race', 'marital_status', 
        'ethnicity', 'currentzip_medianincome_2010', 'public_payer', 
        'visit_count', 'notes_ct', 'icd_first_sdate', 'cpt_first_sdate',
        'biobank_genotyped'
    ]
    df_final = df.merge(df_demo[demo_cols], on='subject_num', how='left')

    df_final['index_date'] = pd.to_datetime(df_final.index_date)
    df_final['sbirth_date'] = pd.to_datetime(df_final.sbirth_date)
    df_final['icd_first_sdate'] = pd.to_datetime(df_final.icd_first_sdate)
    df_final['cpt_first_sdate'] = pd.to_datetime(df_final.cpt_first_sdate)
    df_final = df_final.rename(columns={'sbirth_date': 'birth_date'})

    return df_final


def data_sampling(random_state=42): 
    BASE_data_path = os.path.join(DATA_DIR, args.cohort.upper(), 'base_data')
    DATA_path = os.path.join(DATA_DIR, args.cohort.upper())

    logger.info("Loading base data")
    df = load_pickle(os.path.join(BASE_data_path, "RPDRml__ALL_base_v2024.2.pkl"))
    print(f"    Number of {args.cohort} cases = {len(df[df['label']==1])}")
    print(f"    Number of {args.cohort} controls = {len(df[df['label']==0])}")

    # logger.info("Filtering patients #1")
    # def diff_funct(row):
    #     diff = row['all_dates'][-1]-row['all_dates'][0]
    #     return diff.days
    # df['ehr_duration'] = df.apply(lambda row: diff_funct(row), axis=1)
    # df = df[df['ehr_duration']>=365]
    # print(f"    Number of {args.cohort} cases = {len(df[df['label']==1])}")
    # print(f"    Number of {args.cohort} controls = {len(df[df['label']==0])}")

    logger.info(f"Sampling patient EHR history with {args.lookback_window} year(s) lookback window")
    df_new = common.ehr_sampling(
        df,
        lookback_window=args.lookback_window, 
        random_state=random_state
    )

    print(f"\n{len(df_new)} after data sampling")
    df_cas = df_new[df_new['label']==1]
    df_ctl = df_new[df_new['label']==0]
    print(f"    Number of {args.cohort} cases = {len(df_cas)}")
    print(f"    Number of {args.cohort} controls = {len(df_ctl)}")

    logger.info("Filtering patients by age")
    def diff_funct(row):
        diff = row['all_dates'][-1]-row['all_dates'][0]
        return diff.days
    df_new['ehr_duration'] = df_new.apply(lambda row: diff_funct(row), axis=1)
    # df_new = df_new[df_new['ehr_duration']>=365] # Require min. 1 year of EHR history
    # df_new = df_new[df_new['dx_codes'].str.len() >= 2] # Require min. 1 visits 

    merge_cols = [
        'subject_num', 'gender', 'race', 'marital_status', 'ethnicity',
        'currentzip_medianincome_2010', 'public_payer', 
        # 'ehr_duration', 'icd_first_sdate', 'cpt_first_sdate'
    ]
    df_new = pd.merge(df_new, df[merge_cols], on='subject_num')
    df_new['age'] = df_new.apply(lambda x: common.calc_date_diff(x['pred_date'], x['birth_date']), axis=1)
    df_new = df_new[df_new['age']>=18].reset_index(drop=True) # Require a minimum age of 18 for all patients

    df_new = df_new[df_new['dx_codes'].str.len() > 0].reset_index(drop=True)

    print(f"\n{len(df_new)} after data filtering")
    df_cas = df_new[df_new['label']==1]
    df_ctl = df_new[df_new['label']==0]
    print(f"    Number of {args.cohort} cases = {len(df_cas)}")
    print(f"    Number of {args.cohort} controls = {len(df_ctl)}")

    # Re-balancing case and control ratio
    # Replace random downsampling with exact year matching by prediction year (landmark_year) and demographic vars
    df_matched = common.match_by_demographics(
        df_new,
        label_col='label',
        date_col='pred_date',
        demographic_cols={
            'age': 'age',
            'gender': 'gender',
            'race': 'race',
            'ethnicity': 'ethnicity'
        },
        age_bins=feat_params['custom_bins']['age'],
        matching_ratio=args.matching_ratio,  # e.g., 1 for 1:1 matching
        random_state=random_state
    )
    cohort_pids = set(df_matched['subject_num'].unique())
    print(f"\n{len(cohort_pids)} patients after matching and balancing the cohort")
    df_cas = df_matched[df_matched['label']==1]
    df_ctl = df_matched[df_matched['label']==0]
    print(f"    Number of {args.cohort} cases = {len(df_cas)}")
    print(f"    Number of {args.cohort} controls = {len(df_ctl)}")

    # # Re-balancing case and control ratio
    # df_cases = df_new[df_new['label'] == 1]
    # df_controls = df_new[df_new['label'] == 0]
    # df_controls_sampled = df_controls.sample(n=len(df_cases), random_state=random_state)
    # df_new = pd.concat([df_cases, df_controls_sampled], ignore_index=True)
    # df_new = df_new.sample(frac=1, random_state=random_state).reset_index(drop=True)
    # cohort_pids = set(df_new['subject_num'].unique())
    # print(f"\n{len(cohort_pids)} patients after balancing the cohort (again)")
    # print(df_new['label'].value_counts())

    print(f"\nAverage age for cases = {np.mean(df_cas['age'])}")
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
    
    df_matched = df_matched.drop(['birth_date'], axis=1)
    df_matched['label'] = df_matched['label'].astype(int)

    logger.info("Adding more healthcare utilization features")
    df_final = common.add_hucounts(df_matched)

    logger.info("Saving sampled data frame in pickle format")
    output_file = f"RPDRml__{args.cohort.upper()}_all.{str(args.lookback_window)}y.pkl"
    save_pickle(df_final, os.path.join(DATA_path, args.data_folder, output_file))

    logger.info("Splitting data into training and testing sets")
    df_train, df_temp = common.data_split( #70/15/15
        df_final, test_size=0.30, 
        col='label', random_seed=random_state
    )
    df_val, df_test = common.data_split(
        df_temp, test_size=0.50, 
        col='label', random_seed=random_state
    )

    print("\nTrain data")
    print(len(df_train))
    print(df_train.label.value_counts())
    print()
    print("Validation data")
    print(len(df_val))
    print(df_val.label.value_counts())
    print("Test data")
    print(len(df_test))
    print(df_test.label.value_counts())
    print()

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_data_file = f"RPDRml__{args.cohort.upper()}_train.{str(args.lookback_window)}y.pk"
    val_data_file = f"RPDRml__{args.cohort.upper()}_val.{str(args.lookback_window)}y.pk"
    test_data_file = f"RPDRml__{args.cohort.upper()}_test.{str(args.lookback_window)}y.pk"
    save_pickle(df_train, os.path.join(DATA_path, args.data_folder, train_data_file))
    save_pickle(df_val, os.path.join(DATA_path, args.data_folder, val_data_file))
    save_pickle(df_test, os.path.join(DATA_path, args.data_folder, test_data_file))
    print("All saved")


def main():
    # logger.info("Create base data")
    # create_base_data(base_data_dict)

    logger.info("Data sampling")
    data_sampling(random_state)

    print(f"Overall took {(time.time()-t0)/60} mins.") 


if __name__ == "__main__":
    main()