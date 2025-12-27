import numpy as np
import pandas as pd
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Tuple, List, Union, Dict, Optional
from sklearn.preprocessing import MaxAbsScaler

from config import feat_params
import features.feature_utils as util


CODING_FEATURES = {'dx', 'nlp', 'med', 'proc', 'lab', 'all'}
DEMOGRAPHIC_FEATURE = 'demo'
ALL_VALID_FEATURES = CODING_FEATURES | {DEMOGRAPHIC_FEATURE}


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    feat_type: str
    feat_transform: str = 'bow'
    feat_lvl: str = 'patient'
    feat_scaling: bool = False
    max_df: float = 1.0
    min_df: int = 1
    ngram_range: Tuple[int, int] = (1, 1)


def compute_demo_features(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame, 
        binning: bool = True,
        custom_bins: Optional[dict] = None
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """ Extract demographic features from patient dataframes. """

    # Define feature columns
    categorical_cols = [
        'gender', 'race', 'ethnicity'
    ]
    continuous_cols = [
        'age', 
    ]

    # Initialize custom_bins
    if custom_bins is None:
        custom_bins = feat_params.get('custom_bins', {}).copy()
    else:
        custom_bins = custom_bins.copy()
    
    # Set up age bins: 0-25, 25-40, 40-60, 60-75, 75+
    if 'age' not in custom_bins:
        custom_bins['age'] = [0, 25, 40, 60, 75, float('inf')]

    train_features = []
    test_features = []
    
    # Handle categorical features
    for col in categorical_cols:
        train_encoded, test_encoded = util.encode_categorical(
            df_train[col], 
            df_test[col], 
            col_name=col
        )
        train_features.append(train_encoded)
        test_features.append(test_encoded)

    # Handle continuous features
    if binning:
        # Compute bins from training data only
        bin_edges = {}
        for col in continuous_cols:
            if col in custom_bins:
                bin_edges[col] = custom_bins[col] # Use predefined bins
            else:
                # Compute quantile bins from training data
                col_name, edges = util.compute_bin_edges(df_train[col], q=5)
                bin_edges[col_name] = edges
        
        # Apply binning
        for col in continuous_cols:
            if col in custom_bins:
                # Use custom binning function for predefined bins
                train_binned = util.apply_custom_bins(
                    df_train[col], 
                    bin_edges[col], 
                    col_name=col
                )
                test_binned = util.apply_custom_bins(
                    df_test[col], 
                    bin_edges[col], 
                    col_name=col
                )
            else:
                # Use quantile binning
                train_binned = util.smart_qcut(
                    df_train[col], 
                    bin_edges[col], 
                    use_ordinal_encoding=False
                )
                test_binned = util.smart_qcut(
                    df_test[col], 
                    bin_edges[col], 
                    use_ordinal_encoding=False
                )
            train_features.append(train_binned)
            test_features.append(test_binned)
    else:
        # Use continuous features as is
        train_features.append(df_train[continuous_cols])
        test_features.append(df_test[continuous_cols])
    
    # Combine all features
    df_train_final = pd.concat(train_features, axis=1)
    df_test_final = pd.concat(test_features, axis=1)
    df_test_final = df_test_final.reindex(columns=df_train_final.columns, fill_value=0)
    
    # Convert to sparse matrices
    X_train = sp.csr_matrix(df_train_final.values.astype(float))
    X_test = sp.csr_matrix(df_test_final.values.astype(float))
    feature_names = df_train_final.columns.values
    
    return X_train, X_test, feature_names


def compute_coding_features(df_train, 
                        df_test, 
                        feat_lvl,
                        feat_type,
                        max_df, 
                        min_df, 
                        ngram_range):
    """
    Extract medical code features from patient dataframes.
    
    Code types used:
    - dx: Phecode
    - med: RXNORM
    - proc: SNOMED
    """

    if feat_lvl == 'patient':
        data_train = [[" ".join(list(set([code for code in visit]))) for visit in pt]
                                        for pt in df_train[feat_type+"_codes"]]
        data_test = [[" ".join(list(set([code for code in visit]))) for visit in pt]
                                        for pt in df_test[feat_type+"_codes"]]
    elif feat_lvl == 'visit':
        data_bow = [" ".join([code for visit in pt for code in visit]) 
                                        for pt in df_train[feat_type+"_codes"]]
        data_train = [[" ".join([code for code in visit]) for visit in pt]
                                        for pt in df_train[feat_type+"_codes"]]
        data_test = [[" ".join([code for code in visit]) for visit in pt]
                                        for pt in df_test[feat_type+"_codes"]]
    else:
        raise ValueError("Wrong arguments: feat_lvl")

    if feat_lvl == 'visit':
        model = util.ComputeBowFeatures(
            max_df=max_df, min_df=min_df, ngram_range=ngram_range, binary=True
        )
        model.fit(data_bow)
        X_train = [model.transform(pt) for pt in data_train]
        X_test = [model.transform(pt) for pt in data_test]
    elif feat_lvl == 'patient':
        data_train = [" ".join(pt) for pt in data_train]
        data_test = [" ".join(pt) for pt in data_test]

        model = util.ComputeBowFeatures(
            max_df=max_df, min_df=min_df, ngram_range=ngram_range, binary=False
        )
        model.fit(data_train)
        X_train = model.transform(data_train)
        X_test = model.transform(data_test)
    else:
        raise ValueError("Wrong arguments: feat_lvl")

    return X_train, X_test, model.feature_names_


def _compute_single_feature(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feat_type: str,
        config: FeatureConfig
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Compute features for a single feature type."""

    if feat_type == DEMOGRAPHIC_FEATURE:
        print("Computing demographics features")
        return compute_demo_features(df_train, df_test, feat_params['custom_bins'])
    
    elif feat_type in CODING_FEATURES:
        print(f"Computing {feat_type.upper()} BOW features per {config.feat_lvl}")
        return compute_coding_features(
            df_train, df_test,
            config.feat_lvl,
            feat_type,
            config.max_df,
            config.min_df,
            config.ngram_range
        )
    
    else:
        raise ValueError(f"Unknown feature type: {feat_type}")


def _compute_multiple_features(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feat_types: List[str],
        config: FeatureConfig
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Compute and combine multiple feature types."""

    display_names = [ft.upper() for ft in feat_types]
    print(f"Computing {' and '.join(display_names)} BOW features per {config.feat_lvl}")

    # Separate demo from coding features
    has_demo = DEMOGRAPHIC_FEATURE in feat_types
    coding_types = [ft for ft in feat_types if ft != DEMOGRAPHIC_FEATURE]

    if len(coding_types) > 1:
        coding_feat_type = '_'.join(coding_types)
    elif len(coding_types) == 1:
        coding_feat_type = coding_types[0]
    else:
        coding_feat_type = None

    # Compute demographics if present
    X_train_cov = X_test_cov = cov_names = None
    if has_demo:
        X_train_cov, X_test_cov, cov_names = compute_demo_features(df_train, df_test, feat_params['custom_bins'])

    # Compute coding features
    if coding_feat_type:
        X_train_obs, X_test_obs, obs_names = compute_coding_features(
            df_train, df_test,
            config.feat_lvl,
            coding_feat_type,
            config.max_df,
            config.min_df,
            config.ngram_range
        )

    # Combine features based on level
    if config.feat_lvl == 'visit':
        # Concatenate cov and obs features per visit
        if has_demo and coding_feat_type:
            feature_names = np.hstack([cov_names, obs_names])
            X_train = [sp.vstack([sp.hstack([X_train_cov[i], X_train_obs[i][j]]) 
                                  for j in range(X_train_obs[i].shape[0])]) 
                                    for i in range(len(X_train_obs))]
            X_test = [sp.vstack([sp.hstack([X_test_cov[i], X_test_obs[i][j]]) 
                                  for j in range(X_test_obs[i].shape[0])]) 
                                    for i in range(len(X_test_obs))]
        elif coding_feat_type:
            X_train = X_train_obs
            X_test = X_test_obs
            feature_names = obs_names
        else:
            X_train = X_train_cov
            X_test = X_test_cov
            feature_names = cov_names
    else:
        # Patient-level
        if has_demo and coding_feat_type:
            X_train = sp.hstack([X_train_cov, X_train_obs])
            X_test = sp.hstack([X_test_cov, X_test_obs])
            feature_names = np.hstack([cov_names, obs_names])
        elif coding_feat_type:
            X_train = X_train_obs
            X_test = X_test_obs
            feature_names = obs_names
        else:
            X_train = X_train_cov
            X_test = X_test_cov
            feature_names = cov_names
    
    return X_train, X_test, feature_names


def compute_features(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        feat_type: str = 'demo_dx',
        feat_transform: str = 'bow',
        feat_lvl: str = 'patient',
        feat_scaling: bool = False,
        max_df: float = 1.0,
        min_df: int = 1,
        ngram_range: Tuple[int, int] = (1, 1)
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """ Compute features from patient dataframes. """

    config = FeatureConfig(
        feat_type=feat_type,
        feat_transform=feat_transform,
        feat_lvl=feat_lvl,
        feat_scaling=feat_scaling,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range
    )

    # Parse feature types
    feat_types = feat_type.split("_")
    
    # Compute features
    if len(feat_types) == 1:
        X_train, X_test, feature_names = _compute_single_feature(
            df_train, df_test, feat_types[0], config
        )
    else:
        X_train, X_test, feature_names = _compute_multiple_features(
            df_train, df_test, feat_types, config
        )

    if (feat_lvl == 'patient'):
        if feat_scaling:
            print("Scaling features")
            scaler = MaxAbsScaler()
            X_train = X_train = scaler.fit_transform(X_train)
            X_test = X_test = scaler.transform(X_test)

        print(f"Feature size = {X_train.shape[1]} dimensions")
        return X_train, X_test, feature_names
    
    elif feat_lvl == 'visit':
        print(f"Feature size = {X_train[0].shape[1]} dimensions")
        return X_train, X_test, feature_names
    
    else:
        raise ValueError("Wrong arguments: feat_lvl")
    
