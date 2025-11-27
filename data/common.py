"""
Utility functions for data preparation
"""
import math
import random
import datetime
import numpy as np
import pandas as pd
from typing import Any
from collections import defaultdict
from sklearn.model_selection import train_test_split


def calc_date_diff(d1, d2):
    d1 = pd.to_datetime(d1)
    d2 = pd.to_datetime(d2)
    age = d1.year - d2.year
    age -= ((d1.month, d1.day) < (d2.month, d2.day))
    return age


def extract_dates(x, dx_col='MDD_DX', date_col='sstart_date_dx'):   
    index_date = x.loc[x[dx_col]==1, date_col].min()
    return index_date


def deduplicate_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def is_nan(value: Any) -> bool:
    """Check if a value is NaN or None."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ('nan', 'none', ''):
        return True
    return False


def deduplicate_dates_with_codes(dates, codes):
    """
    Deduplicate dates and remove entries where codes contain only NaN values.
    """
    if not dates or not codes:
        return [], []
    if is_nan(dates) or is_nan(codes):
        return [], []
    
    dates = deduplicate_order(dates) # Deduplicate the dates list
    if len(dates) != len(codes):
        raise ValueError("dates and codes must have the same length")
    
    final_dates = []
    final_codes = []
    for date, code_list in zip(dates, codes):
        if not code_list:
            continue

        code_list = [code for code in code_list if not is_nan(code)]

        if code_list:
            final_dates.append(date)
            final_codes.append(code_list)

    return final_dates, final_codes


def filter_cohort(df_cohort, cutoff_date="2017-01-01"):
    df_cohort['index_date'] = pd.to_datetime(df_cohort['index_date'], format='%Y-%m-%d', errors='coerce')
    cutoff_date = pd.to_datetime(cutoff_date, format='%Y-%m-%d')

    # Count before filtering
    n_controls = len(df_cohort[df_cohort['label'] == 0])
    n_cases_before = len(df_cohort[df_cohort['label'] == 1])

    # Filter: keep all controls and cases >= cutoff
    mask = (df_cohort['label'] == 0) | (
        (df_cohort['label'] == 1) & (df_cohort['index_date'] >= cutoff_date)
    )
    df_filtered = df_cohort[mask].copy()

    # Count after filtering
    n_cases_after = len(df_filtered[df_filtered['label'] == 1])
    
    print(f"Number of controls (all retained): {n_controls}")
    print(f"Number of cases before filtering: {n_cases_before}")
    print(f"Number of cases after filtering (index dates >= {cutoff_date}): {n_cases_after}")
    print(f"Number of cases excluded: {n_cases_before - n_cases_after}")
    
    cohort_pids = df_filtered['subject_num'].unique().tolist()
    print(f"Total cohort size: {len(cohort_pids)}\n")
    
    return df_filtered, cohort_pids


def groupby_pid(dx, df_cohort, data_type, cutoff_date=None):
    """ Group data by patient ID """
    dx['sstart_date'] = pd.to_datetime(dx.sstart_date, errors='coerce')
    if cutoff_date: # Only use data >= the cut-off date
        dx = dx[dx['sstart_date'] >= cutoff_date]
    dx = dx.sort_values(['subject_num', 'sstart_date'], ascending=True)
    dx = dx.loc[pd.notnull(dx['sstart_date'])]
    
    # Merge with patient info
    dx = dx.merge(df_cohort[['subject_num', 'index_date', 'label']], 
                  on='subject_num', 
                  how='inner')  # Use 'left' if you want to keep all patients
    
    # Group and aggregate
    dx_dict = []
    for pid, df in dx.groupby('subject_num'):
        if data_type == "LAB":
            concept_codes = [i+"|"+j for i, j in zip(df.concept_code.tolist(), df.valueflag.tolist())]
        else:
            concept_codes = df.concept_code.tolist()
            
        dx_dict.append({
            "subject_num": pid,
            # "encounter_num": df.encounter_num.tolist(),
            "concept_codes": concept_codes,
            "start_dates": df.sstart_date.tolist(),
            "label": df['label'].iloc[0],  
            "index_date": df['index_date'].iloc[0]
        })
    
    return pd.DataFrame.from_dict(dx_dict)


def group_by_dates(concepts, dates):
    if not concepts:  
        return []
    
    # Create a mapping of date -> list of (index, concept) pairs
    date_to_items = defaultdict(list)
    for idx, (concept, date) in enumerate(zip(concepts, dates)):
        date_to_items[date].append((idx, concept))
    
    # Build result preserving first occurrence order
    seen = set()
    result = []
    for date in dates:
        if date not in seen:
            seen.add(date)
            # Extract concepts for this date, preserving their relative order
            concepts_for_date = [concept for _, concept in sorted(date_to_items[date])]
            result.append(concepts_for_date)
    
    return result


def group_concepts(df):
    """ Group patient obs data by dates """
    df['concepts_by_date'] = df.apply(
        lambda row: group_by_dates(row['concept_codes'], row['start_dates']), 
        axis=1
    )
    return df


def balance_case_control_data(df_dx, df_med, df_lab, df_proc, mgbb_pids, random_state=42):
    """
    Balance case and control numbers across multiple dataframes.
    """    
    # Exclude MGBB participants
    df_dx_filtered = df_dx[~df_dx['subject_num'].isin(mgbb_pids)]
    print(f"\n{len(df_dx_filtered)} patients after excluding MGBB participants")
    
    # Separate cases and controls
    df_cases = df_dx_filtered[df_dx_filtered['label'] == 1]
    df_controls = df_dx_filtered[df_dx_filtered['label'] == 0]
    
    n_cases = len(df_cases)
    n_controls = len(df_controls)
    print(f"    Including {n_cases} cases and {n_controls} controls")
    
    # # Validate we have enough controls
    # if n_controls < n_cases:
    #     print(f"Insufficient controls ({n_controls}) for {n_cases} cases. ")
    #     df_controls_sampled = df_controls
    # else:
    #     df_controls_sampled = df_controls.sample(n=n_cases, random_state=random_state)
    
    # # Combine and shuffle
    # df_dx_balanced = pd.concat([df_cases, df_controls_sampled], ignore_index=True)
    # df_dx_balanced = df_dx_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_dx_balanced = df_dx_filtered
    
    cohort_pids = set(df_dx_balanced['subject_num'].unique())
    print(f"\n{len(cohort_pids)} patients after balancing the cohort")
    print(df_dx_balanced['label'].value_counts())
    
    # Filter other dataframes
    print(f"    {len(df_dx_balanced)} patients in DX after sampling")
    
    df_med_balanced = df_med[df_med['subject_num'].isin(cohort_pids)]
    print(f"    {len(df_med_balanced)} patients in MED after sampling")
    
    df_lab_balanced = df_lab[df_lab['subject_num'].isin(cohort_pids)]
    print(f"    {len(df_lab_balanced)} patients in LAB after sampling")
    
    df_proc_balanced = df_proc[df_proc['subject_num'].isin(cohort_pids)]
    print(f"    {len(df_proc_balanced)} patients in PROC after sampling\n")
    
    return df_dx_balanced, df_med_balanced, df_lab_balanced, df_proc_balanced


def randomly_sample_control_visit(all_dates):
    """
    Randomly sample a visit date for controls to use as reference point.
    """
    if not all_dates:
        return None
        
    # Sort dates to ensure we have enough history for lookback
    dates_arr = np.array(all_dates)
    sorted_dates = np.sort(dates_arr)
    min_date = sorted_dates[0]
    
    # Find eligible dates (those with at least 1 years of history)
    min_required_date = min_date + pd.DateOffset(years=1)
    eligible_dates = sorted_dates[sorted_dates >= min_required_date]
    # eligible_dates = sorted_dates[2:]
    
    if len(eligible_dates) == 0:
        # If no dates have enough history, just use the last date
        return pd.to_datetime(sorted_dates[-1])
    
    # Randomly sample from eligible dates
    sampled_date = np.random.choice(eligible_dates)
    return pd.to_datetime(sampled_date)


def find_case_pred_visit(all_dates, index_date, min_days_before=90):
    """
    Find the last visit that is min_days_before prior to the index date.
    """
    if not all_dates:
        return None
    
    dates_arr = np.array(all_dates)

    # Find dates that are at least min_days_before the index_date
    days_before = (index_date - dates_arr).astype('timedelta64[D]').astype(int)
    eligible_mask = days_before >= min_days_before
    eligible_dates = dates_arr[eligible_mask]
    
    if len(eligible_dates) == 0:
        return None
    
    # Return the most recent eligible date
    return pd.to_datetime(np.max(eligible_dates))


def lookback_sampling(dates, codes, lookback_date, pred_date):
    if not dates or not codes:
        return [], []
    
    if len(dates) != len(codes):
        raise ValueError(f"Dates and codes must have same length. Got {len(dates)} dates and {len(codes)} codes.")
    
    dates_arr = np.array(dates)
    # Check if arrays contain only NaN values
    if pd.isna(dates_arr).all():
        return [], []

    # Filter for lookback window
    mask = (dates_arr >= lookback_date) & (dates_arr <= pred_date)

    # Get indices where mask is True
    valid_indices = np.where(mask)[0]
    
    # Filter both dates and codes using valid indices
    filtered_dates = [dates[i] for i in valid_indices]
    filtered_codes = [codes[i] for i in valid_indices]
    
    return filtered_dates, filtered_codes


def ehr_sampling(df, lookback_window=1, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)

    def check_ehr_duration(all_dates, pred_date, min_days=365):
        """
        Check if there's sufficient EHR history before prediction date
        """
        if not all_dates or len(all_dates) == 0:
            return False
        
        first_ehr_date = all_dates[0] # dates are sorted already
        ehr_duration_days = (pred_date - first_ehr_date).days
        return ehr_duration_days >= min_days

    def process_row(row, lookback_window):
        """
        Process a single patient (row of EHR dataframe)
        """
        index_date = row['index_date']

        # Calculate lookback date
        if pd.isnull(index_date):
            # Controls: randomly sample a visit date as reference
            pred_date = randomly_sample_control_visit(row['all_dates'])
            # pred_date = row['all_dates'][-1]
            if pred_date is None:
                return None
        else: 
            # Cases: lookback from index date minus 90 days
            pred_date = find_case_pred_visit(row['all_dates'], index_date, min_days_before=90)
            if not pred_date:
                return None
            
        # Filter by EHR length: Check for minimum EHR history (1 year)
        if not check_ehr_duration(row['all_dates'], pred_date, min_days=365):
            return None
        
        lookback_date = pred_date - pd.DateOffset(years=lookback_window)
        
        # Apply lookback sampling to all date/code pairs
        all_dates, all_codes = lookback_sampling(
            row['all_dates'], row['all_codes'], lookback_date, pred_date
        )
        
        # Skip if no valid data
        if not all_dates or not all_codes:
            return None
            
        # Apply lookback sampling to each data type
        dx_dates, dx_codes = lookback_sampling(
            row['dx_dates'], row['dx_codes'], lookback_date, pred_date)
        med_dates, med_codes = lookback_sampling(
            row['med_dates'], row['med_codes'], lookback_date, pred_date)
        lab_dates, lab_codes = lookback_sampling(
            row['lab_dates'], row['lab_codes'], lookback_date, pred_date)
        proc_dates, proc_codes = lookback_sampling(
            row['proc_dates'], row['proc_codes'], lookback_date, pred_date)
        
        # if (len(all_codes) <=2) or (len(dx_codes)==0):
        #     return None
    
        return pd.Series({
            'subject_num': row['subject_num'],
            'pred_date': pred_date,
            'index_date': index_date,
            'birth_date': row['birth_date'],
            'all_dates': all_dates,
            'all_codes': all_codes,
            'dx_dates': dx_dates,
            'dx_codes': dx_codes,
            'med_dates': med_dates,
            'med_codes': med_codes,
            'lab_dates': lab_dates,
            'lab_codes': lab_codes,
            'proc_dates': proc_dates,
            'proc_codes': proc_codes,
            'label': row['label'],
        })
    
    df_result = df.apply(
        lambda row: process_row(row, lookback_window), 
        axis=1
    )
    df_result = df_result.dropna(how='all')

    if isinstance(df_result, pd.Series):
        df_result = pd.DataFrame(df_result.tolist())

    df_result = df_result.reset_index(drop=True)
    
    return df_result


def add_hucounts(df):
    """Add count-based healthcare utilization features"""
    def calculate_all_counts(row):
        dx_count = sum(
            sum(1 for code in codes if 'PheCode:' in str(code))
            for codes in (row['dx_codes'] or [])
        )
        
        med_count = sum(
            sum(1 for code in codes if 'RXNORM:' in str(code))
            for codes in (row['med_codes'] or [])
        )
        
        proc_count = sum(
            sum(1 for code in codes if 'CPT4:' in str(code))
            for codes in (row['proc_codes'] or [])
        )
        
        # Visit count == number of days of having any code
        visit_count = len(row['all_dates']) if row['all_dates'] else 0
        
        return pd.Series({
            'dx_count': dx_count,
            'med_count': med_count,
            'proc_count': proc_count,
            'visit_count': visit_count
        })
    
    # Calculate all counts at once
    counts_df = df.apply(calculate_all_counts, axis=1)
    
    return pd.concat([df, counts_df], axis=1)


def data_split(df, test_size, col, random_seed):
    df_train, df_test, _, _ = train_test_split(
        df, 
        df[col], 
        test_size=test_size, 
        shuffle=True,
        stratify=df[col],
        random_state=random_seed
    )
    return df_train, df_test

    
def match_by_year(df: pd.DataFrame,
                 label_col: str = "label",
                 date_col: str = "pred_date", 
                 case_label: int = 1,
                 control_label: int = 0,
                 matching_ratio: int = 1,
                 random_state: int = 42) -> pd.DataFrame:
    '''
    Exact year matching: each case is matched to control(s) from the same prediction year.
    '''
    np.random.seed(random_state)
    
    # Extract year efficiently
    df = df.copy()
    df['landmark_year'] = pd.to_datetime(df[date_col]).dt.year
    years = df['landmark_year'].values
    labels = df[label_col].values
    
    # Boolean masks for cases and controls
    case_mask = labels == case_label
    control_mask = labels == control_label
    
    # Original counts
    n_original_cases = case_mask.sum()
    n_original_controls = control_mask.sum()
    
    # Get unique years from cases
    case_years = years[case_mask]
    unique_years = np.unique(case_years)
    
    # Track matched and unmatched
    matched_case_indices = []
    matched_control_indices = []
    unmatched_case_indices = []
    
    # Year-by-year summary for detailed reporting
    year_summary = []
    
    for year in unique_years:
        # Get indices for this year
        year_case_indices = np.where((years == year) & case_mask)[0]
        year_control_indices = np.where((years == year) & control_mask)[0]
        
        n_cases = len(year_case_indices)
        n_controls = len(year_control_indices)
        
        if n_controls == 0:
            # No controls available for this year
            unmatched_case_indices.extend(year_case_indices.tolist())
            year_summary.append({
                'year': year,
                'cases': n_cases,
                'controls_available': 0,
                'cases_matched': 0,
                'controls_matched': 0,
                'cases_unmatched': n_cases
            })
            continue
            
        # Determine how many cases we can match
        n_matchable = min(n_cases, n_controls // matching_ratio)
        
        if n_matchable == 0:
            # Not enough controls for even one case
            unmatched_case_indices.extend(year_case_indices.tolist())
            year_summary.append({
                'year': year,
                'cases': n_cases,
                'controls_available': n_controls,
                'cases_matched': 0,
                'controls_matched': 0,
                'cases_unmatched': n_cases
            })
            continue
        
        # Randomly select cases and controls
        if n_matchable < n_cases:
            # Can't match all cases - need to select which ones to match
            selected_case_idx = np.random.choice(year_case_indices, size=n_matchable, replace=False)
            # Track unmatched cases
            unmatched = np.setdiff1d(year_case_indices, selected_case_idx)
            unmatched_case_indices.extend(unmatched.tolist())
        else:
            # Can match all cases
            selected_case_idx = year_case_indices
        
        selected_control_idx = np.random.choice(
            year_control_indices, 
            size=n_matchable * matching_ratio, 
            replace=False
        )
        
        matched_case_indices.extend(selected_case_idx.tolist())
        matched_control_indices.extend(selected_control_idx.tolist())
        
        year_summary.append({
            'year': year,
            'cases': n_cases,
            'controls_available': n_controls,
            'cases_matched': len(selected_case_idx),
            'controls_matched': len(selected_control_idx),
            'cases_unmatched': n_cases - len(selected_case_idx)
        })
    
    # Combine all matched indices
    all_matched_indices = matched_case_indices + matched_control_indices
    
    # Extract matched rows
    if all_matched_indices:
        result = df.iloc[all_matched_indices].copy()
        # Shuffle
        result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        result = pd.DataFrame()
    
    # Calculate summary statistics
    n_matched_cases = len(matched_case_indices)
    n_matched_controls = len(matched_control_indices)
    n_unmatched_cases = len(unmatched_case_indices)
    
    # Report matching summary
    print("\n" + "="*50)
    print("MATCHING SUMMARY")
    print("="*50)
    print(f"Original cases: {n_original_cases:,}")
    print(f"Original controls: {n_original_controls:,}")
    print(f"Matched cases: {n_matched_cases:,}")
    print(f"Matched controls: {n_matched_controls:,}")
    print(f"Unmatched cases: {n_unmatched_cases:,}")
    
    if n_matched_cases > 0:
        print(f"Achieved matching ratio: 1:{n_matched_controls/n_matched_cases:.2f}")
        print(f"Match rate: {n_matched_cases/n_original_cases*100:.1f}% of cases matched")
    
    return result


def match_by_demographics(df: pd.DataFrame,
                         label_col: str = "label",
                         date_col: str = "pred_date",
                         demographic_cols: dict = None,
                         age_bins: list = None,
                         case_label: int = 1,
                         control_label: int = 0,
                         matching_ratio: int = 1,
                         random_state: int = 42) -> pd.DataFrame:
    '''
    Match cases to controls by prediction year and demographic variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with cases and controls
    label_col : str
        Column name for case/control label
    date_col : str
        Column name for prediction date
    demographic_cols : dict
        Dictionary mapping demographic variable names to column names.
        Example: {'age': 'age', 'gender': 'gender', 'race': 'race', 'ethnicity': 'ethnicity'}
        If None, uses default names
    age_bins : list
        Bin edges for age grouping. Example: [0, 18, 25, 40, 60, 75, float('inf')]
        If None, uses default bins
    case_label : int
        Value indicating cases
    control_label : int
        Value indicating controls
    matching_ratio : int
        Number of controls per case (e.g., 1 for 1:1, 2 for 1:2)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Matched dataset with cases and controls
    '''
    np.random.seed(random_state)
    
    # Set defaults
    if demographic_cols is None:
        demographic_cols = {
            'age': 'age',
            'gender': 'gender', 
            'race': 'race',
            'ethnicity': 'ethnicity'
        }
    
    if age_bins is None:
        age_bins = [0, 18, 25, 40, 60, 75, float('inf')]
    
    # Create working copy
    df = df.copy()
    
    # Extract year
    df['_match_year'] = pd.to_datetime(df[date_col]).dt.year
    
    # Bin age if age column exists
    if 'age' in demographic_cols:
        age_col = demographic_cols['age']
        age_labels = [f"{int(age_bins[i])}-{int(age_bins[i+1]) if age_bins[i+1] != float('inf') else '+'}" 
                     for i in range(len(age_bins)-1)]
        df['_match_age_bin'] = pd.cut(df[age_col], bins=age_bins, labels=age_labels, right=False)
    
    # Create stratification columns
    strata_cols = ['_match_year']
    if 'age' in demographic_cols:
        strata_cols.append('_match_age_bin')
    
    # Add other demographic columns to stratification
    for demo_name, demo_col in demographic_cols.items():
        if demo_name != 'age':  # Age already handled
            if demo_col in df.columns:
                strata_cols.append(demo_col)
    
    # Create stratification key
    df['_stratum'] = df[strata_cols].astype(str).agg('_'.join, axis=1)
    
    # Boolean masks for cases and controls
    labels = df[label_col].values
    case_mask = labels == case_label
    control_mask = labels == control_label
    
    # Original counts
    n_original_cases = case_mask.sum()
    n_original_controls = control_mask.sum()
    
    # Get unique strata
    unique_strata = df['_stratum'].unique()
    
    # Track matched and unmatched
    matched_case_indices = []
    matched_control_indices = []
    unmatched_case_indices = []
    
    # Stratum-by-stratum summary
    stratum_summary = []
    
    for stratum in unique_strata:
        # Get indices for this stratum
        stratum_mask = df['_stratum'] == stratum
        stratum_case_indices = np.where(stratum_mask & case_mask)[0]
        stratum_control_indices = np.where(stratum_mask & control_mask)[0]
        
        n_cases = len(stratum_case_indices)
        n_controls = len(stratum_control_indices)
        
        # Skip if no cases in this stratum
        if n_cases == 0:
            continue
        
        if n_controls == 0:
            # No controls available for this stratum
            unmatched_case_indices.extend(stratum_case_indices.tolist())
            stratum_summary.append({
                'stratum': stratum,
                'cases': n_cases,
                'controls_available': 0,
                'cases_matched': 0,
                'controls_matched': 0,
                'cases_unmatched': n_cases
            })
            continue
        
        # Determine how many cases we can match
        n_matchable = min(n_cases, n_controls // matching_ratio)
        
        if n_matchable == 0:
            # Not enough controls for even one case
            unmatched_case_indices.extend(stratum_case_indices.tolist())
            stratum_summary.append({
                'stratum': stratum,
                'cases': n_cases,
                'controls_available': n_controls,
                'cases_matched': 0,
                'controls_matched': 0,
                'cases_unmatched': n_cases
            })
            continue
        
        # Randomly select cases and controls
        if n_matchable < n_cases:
            # Can't match all cases - need to select which ones to match
            selected_case_idx = np.random.choice(stratum_case_indices, size=n_matchable, replace=False)
            # Track unmatched cases
            unmatched = np.setdiff1d(stratum_case_indices, selected_case_idx)
            unmatched_case_indices.extend(unmatched.tolist())
        else:
            # Can match all cases
            selected_case_idx = stratum_case_indices
        
        selected_control_idx = np.random.choice(
            stratum_control_indices,
            size=n_matchable * matching_ratio,
            replace=False
        )
        
        matched_case_indices.extend(selected_case_idx.tolist())
        matched_control_indices.extend(selected_control_idx.tolist())
        
        stratum_summary.append({
            'stratum': stratum,
            'cases': n_cases,
            'controls_available': n_controls,
            'cases_matched': len(selected_case_idx),
            'controls_matched': len(selected_control_idx),
            'cases_unmatched': n_cases - len(selected_case_idx)
        })
    
    # Combine all matched indices
    all_matched_indices = matched_case_indices + matched_control_indices
    
    # Extract matched rows and clean up temporary columns
    if all_matched_indices:
        result = df.iloc[all_matched_indices].copy()
        # Drop temporary columns
        result = result.drop(columns=['_match_year', '_stratum'])
        if '_match_age_bin' in result.columns:
            result = result.drop(columns=['_match_age_bin'])
        # Shuffle
        result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        result = pd.DataFrame()
    
    # Calculate summary statistics
    n_matched_cases = len(matched_case_indices)
    n_matched_controls = len(matched_control_indices)
    n_unmatched_cases = len(unmatched_case_indices)
    n_total_strata = len(unique_strata)
    n_strata_with_matches = sum(1 for s in stratum_summary if s['cases_matched'] > 0)
    n_strata_no_controls = sum(1 for s in stratum_summary if s['controls_available'] == 0)
    n_strata_insufficient_controls = sum(1 for s in stratum_summary 
                                         if s['controls_available'] > 0 and s['cases_matched'] < s['cases'])
    
    # Print summary
    print("\n" + "="*60)
    print("MATCHING SUMMARY")
    print("="*60)
    print(f"Matching on: prediction year AND demographic vars: {', '.join([k for k in demographic_cols.keys()])}")
    print(f"\nOriginal dataset:")
    print(f"  Cases: {n_original_cases:,}")
    print(f"  Controls: {n_original_controls:,}")
    print(f"\nMatching results:")
    print(f"  Matched cases: {n_matched_cases:,}")
    print(f"  Matched controls: {n_matched_controls:,}")
    print(f"  Unmatched cases: {n_unmatched_cases:,}")
    
    if n_matched_cases > 0:
        print(f"\nMatching performance:")
        print(f"  Achieved ratio: 1:{n_matched_controls/n_matched_cases:.2f}")
        print(f"  Match rate: {n_matched_cases/n_original_cases*100:.1f}% of cases matched")
    
    print(f"\nStratification details:")
    print(f"  Total unique strata: {n_total_strata:,}")
    print(f"  Strata with matches: {n_strata_with_matches:,}")
    print(f"  Strata with no controls: {n_strata_no_controls:,}")
    print(f"  Strata with insufficient controls: {n_strata_insufficient_controls:,}")
    
    # Show most problematic strata
    if unmatched_case_indices:
        print(f"\nTop 5 strata by unmatched cases:")
        stratum_df = pd.DataFrame(stratum_summary)
        top_unmatched = stratum_df.nlargest(5, 'cases_unmatched')
        for _, row in top_unmatched.iterrows():
            if row['cases_unmatched'] > 0:
                print(f"  {row['stratum']}: {row['cases_unmatched']} unmatched "
                      f"({row['controls_available']} controls available)")
    
    print("="*60 + "\n")
    
    return result