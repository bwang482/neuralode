"""
Code mapping functions
"""

def phecode_mapping(patient_df, dx_dict):
    # Create a clean mapping dictionary (dropping any NaN PheCode codes if needed)
    mapping_df = dx_dict[['concept_code', 'PheCode']].dropna(subset=['PheCode'])
    
    # Merge the patient data with the mapping
    result_df = patient_df.merge(
        mapping_df[['concept_code', 'PheCode']],
        left_on='concept_code',  
        right_on='concept_code',  
        how='left'  # keeps all patient records, even those without a mapping
    )
    
    # Filter out rows where no mapping was found 
    result_df = result_df.dropna(subset=['PheCode'])
    result_df['original_code'] = result_df['concept_code']
    result_df['concept_code'] = result_df['PheCode']
    result_df['concept_code'] = 'PheCode:' + result_df['PheCode'].astype(str) 
    return result_df


def rxnorm_mapping(patient_df, med_dict):
    mapping_df = med_dict[['concept_code', 'RXNORM_ingredient_code']].dropna(subset=['RXNORM_ingredient_code'])
    
    result_df = patient_df.merge(
        mapping_df[['concept_code', 'RXNORM_ingredient_code']],
        left_on='concept_code',  
        right_on='concept_code',  
        how='left'  
    )
    
    result_df = result_df.dropna(subset=['RXNORM_ingredient_code'])
    result_df['original_code'] = result_df['concept_code']
    result_df['concept_code'] = result_df['RXNORM_ingredient_code']
    return result_df


def loinc_mapping(patient_df, lab_dict):
    mapping_df = lab_dict[['concept_id', 'c_loinc']].dropna(subset=['c_loinc'])
    
    result_df = patient_df.merge(
        mapping_df[['concept_id', 'c_loinc']],
        left_on='concept_code',  
        right_on='concept_id',  
        how='left'  
    )
    
    result_df = result_df.dropna(subset=['c_loinc'])
    result_df['original_code'] = result_df['concept_code']
    result_df['concept_code'] = 'LOINC:' + result_df['c_loinc'].astype(str) 
    return result_df


def cpt4_mapping(patient_df):
    result_df = patient_df[patient_df['concept_code'].str.startswith('CPT4')].copy()
    result_df['concept_code'] = result_df['concept_code'].str.replace(r'^(CPT4:)+', 'CPT4:', regex=True)
    return result_df


def snomed_mapping(patient_df, proc_dict):
    result_df = patient_df[patient_df['concept_code'].str.startswith('CPT4')].copy()
    result_df['concept_code'] = result_df['concept_code'].str.replace(r'^(CPT4:)+', 'CPT4:', regex=True)

    result_df = result_df.merge(
        proc_dict[['concept_code', 'concept_id_SNOMED']], 
        on='concept_code', 
        how='left'
    )
    # Rename columns: 'concept_code' contains SNOMED codes while 'cpt_code' has CPT-4 codes
    result_df = result_df.rename(columns={'concept_code': 'cpt_code', 
                                          'concept_id_SNOMED': 'concept_code'})
    return result_df

