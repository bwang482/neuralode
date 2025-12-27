
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Optional
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from utils.general_functs import is_nan, combine_lists


class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown='ignore'):
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        self.model = OneHotEncoder(
            handle_unknown=self.handle_unknown,
        )
        self.model.fit(X)
        self.params_ = self.model.get_params()
        self.feature_names_ = self.model.get_feature_names_out()
        return self

    def transform(self, X):
        return self.model.transform(X)


class FeatureBinning(BaseEstimator, TransformerMixin):
    def __init__(self,  
                n_bins=5, 
                encode='onehot', 
                strategy='quantile'):

        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

    def fit(self, X, y=None):
        self.model = KBinsDiscretizer(
            n_bins=self.n_bins, 
            encode=self.encode, 
            strategy=self.strategy,
        )
        self.model.fit(X)
        self.n_bins_ = self.model.n_bins_
        self.params_ = self.model.get_params()
        self.feature_names_ = self.model.get_feature_names_out()
        return self

    def transform(self, X):
        return self.model.transform(X)


class ComputeBowFeatures(BaseEstimator, TransformerMixin):
    """ Bag of Words feature extractor for medical codes. """
    def __init__(self,  
                max_df=1.0, 
                min_df=1, 
                ngram_range=(1,1),
                binary=True):

        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.binary = binary
        self.model = None
        self.vocab_ = None
        self.params_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        """ Fit the vectorizer on the input data. """
        self.model = CountVectorizer(
            lowercase=False, 
            tokenizer=self.identity_tokenizer,
            token_pattern=None,
            ngram_range=self.ngram_range,
            max_df=self.max_df,
            min_df=self.min_df,
            binary=self.binary,
        )
        self.model.fit(X)
        self.vocab_ = self.model.vocabulary_
        self.params_ = self.model.get_params()
        self.feature_names_ = self.model.get_feature_names_out()
        return self

    def transform(self, X):
        """ Transform the input data to (sparse) feature representation. """
        if self.model is None:
            raise ValueError("Model must be fitted before transform")
        return self.model.transform(X)

    @staticmethod
    def identity_tokenizer(text):
        """ Simple whitespace tokenizer. """
        return text.split()


def make_float(v):
    """ Convert value to float if possible, otherwise return NaN. """
    if v is None or pd.isna(v):
        return np.nan
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def compute_bin_edges(x: pd.Series, q: int = 5) -> Tuple[str, Optional[List[float]]]:
    """ Compute bin edges for quantile-based binning. """
    # Convert to numeric, non-numeric values become NaN
    numeric_series = x.apply(make_float)
    valid_numeric = numeric_series.dropna()
    
    if len(valid_numeric) == 0:
        return (x.name, None)
    
    n_unique = valid_numeric.nunique()
    
    if n_unique <= 1:
        # Not enough unique values to bin
        return (x.name, None)
    elif n_unique == 2:
        # Binary variable, no binning needed
        return (x.name, None)
    else:
        # Compute quantile-based bins
        percentiles = np.linspace(0, 100, q + 1)
        bin_edges = np.unique(np.percentile(valid_numeric, percentiles))
        
        # Need at least 2 edges to create bins
        if len(bin_edges) < 2:
            return (x.name, None)
            
        return (x.name, bin_edges.tolist())


def smart_qcut(
        x: pd.Series, 
        bin_edges: Optional[List[float]], 
        use_ordinal_encoding: bool = False
) -> pd.DataFrame:
    """ Apply binning or encoding to a series based on its characteristics. """
    
    # Convert to numeric where possible
    numeric_series = x.apply(make_float)
    numeric_mask = ~numeric_series.isna()
    
    # Check if we have valid numeric data
    valid_numeric = numeric_series[numeric_mask]
    n_unique = valid_numeric.nunique() if len(valid_numeric) > 0 else 0
    
    if n_unique <= 1 or bin_edges is None:
        # Use one-hot encoding for all values
        return pd.get_dummies(x, prefix=x.name, dummy_na=False)
    
    elif n_unique == 2:
        # Binary numeric variable
        return pd.get_dummies(x, prefix=x.name, dummy_na=False)
    
    else:
        # Apply binning to numeric values
        result = x.copy()
        
        if use_ordinal_encoding:
            # Create ordinal features
            col_names = [f'{x.name}>{edge:.2f}' for edge in bin_edges[:-1]]
            out = pd.DataFrame(0, index=x.index, columns=col_names)
            
            for i, edge in enumerate(bin_edges[:-1]):
                out.loc[numeric_mask, col_names[i]] = (
                    numeric_series[numeric_mask] > edge
                ).astype(int)
            
            # Handle non-numeric values separately
            if (~numeric_mask).any():
                non_numeric_dummies = pd.get_dummies(
                    x[~numeric_mask], 
                    prefix=x.name
                )
                out = pd.concat([out, non_numeric_dummies], axis=1)
            
            return out
        else:
            # Use pd.cut for binning
            result.loc[numeric_mask] = pd.cut(
                valid_numeric,
                bins=bin_edges,
                duplicates='drop',
                include_lowest=True
            )
            
            return pd.get_dummies(result, prefix=x.name, dummy_na=False)


def timebinning(dt: pd.Timestamp, index_dt: pd.Timestamp) -> str:
    DAYS_IN_YEAR = 365
    RECENT_THRESHOLD = 1  # 1 year
    MEDIUM_THRESHOLD = 2  # 2 years
    days_diff = (index_dt - dt).days
    
    if days_diff <= DAYS_IN_YEAR * RECENT_THRESHOLD:
        return "_B1"
    elif days_diff <= DAYS_IN_YEAR * MEDIUM_THRESHOLD:
        return "_B2"
    else:
        return "_B3"


def bp_transform(X):
    home = np.zeros(X.shape[1]).reshape(1, -1)
    X_transformed = np.r_[home, X]
    return X_transformed


def add_time_path(x):
    time_path = np.linspace(0, 1, x.shape[0])
    time_path = time_path.reshape(x.shape[0], 1)
    return np.concatenate((time_path, x), axis=1)


def encode_categorical(
        train_series: pd.Series, 
        test_series: pd.Series, 
        col_name: str,
        drop_first: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Encode categorical variables using one-hot encoding. """
    
    lb = LabelBinarizer()
    lb.fit(train_series)
    X_train = lb.transform(train_series)
    X_test = lb.transform(test_series)

    if len(lb.classes_) == 2: 
        # Binary class case: LabelBinarizer returns 1D array for binary variables
        if drop_first:
            # Keep the single column
            col_names = [f"{col_name}_value_{lb.classes_[1]}"]
        else:
            # Expand to two columns for full one-hot encoding
            X_train = np.column_stack([1 - X_train, X_train])
            X_test = np.column_stack([1 - X_test, X_test])
            col_names = [f"{col_name}_value_{cls}" for cls in lb.classes_]
    else:
        # Multi-class case 
        col_names = [f"{col_name}_value_{cls}" for cls in lb.classes_]
    
    df_train = pd.DataFrame(
        X_train, 
        columns=col_names, 
        index=train_series.index
    )
    df_test = pd.DataFrame(
        X_test, 
        columns=col_names, 
        index=test_series.index
    )
    
    return df_train, df_test


def apply_custom_bins(
        series: pd.Series, 
        bin_edges: List[float], 
        col_name: str
) -> pd.DataFrame:
    """ Apply custom binning to a series using predefined bin edges. """
    # Create bin labels
    labels = []
    for i in range(len(bin_edges) - 1):
        if bin_edges[i+1] == float('inf'):
            labels.append(f"{int(bin_edges[i])}+")
        else:
            labels.append(f"{int(bin_edges[i])}-<{int(bin_edges[i+1])}")
    
    # Apply binning
    binned = pd.cut(
        series, 
        bins=bin_edges, 
        labels=labels,
        include_lowest=True,
        right=False  # Use left-inclusive intervals [a, b)
    )
    
    # Convert to one-hot encoding
    return pd.get_dummies(binned, prefix=col_name)


def combine_df(df):
    combined_dates = []
    combined_codes = []
    for i, row in df.iterrows():
    
        if is_nan(row['dx_codes']):
            row['dx_codes'] = []
        if is_nan(row['dx_dates']):
            row['dx_dates'] = []
        if is_nan(row['med_codes']):
            row['med_codes'] = []
        if is_nan(row['med_dates']):
            row['med_dates'] = []
    
        assert(len(row['dx_codes']) == len(row['dx_dates']))
        assert(len(row['med_codes']) == len(row['med_dates']))
        
        all_dates = row['dx_dates'] + row['med_dates']
        all_codes = row['dx_codes'] + row['med_codes']
    
        combined = combine_lists(all_dates, all_codes)
        combined_dates.append([x[0] for x in combined])
        combined_codes.append([[xi for xl in x[1] for xi in xl] for x in combined])
    
    df['dx_med_codes'] = combined_codes
    df['dx_med_dates'] = combined_dates
    df['dx_med_proc_lab_codes'] = df['all_codes']
    df['dx_med_proc_lab_dates'] = df['all_dates']
    return df


