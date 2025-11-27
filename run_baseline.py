import os
import time
import yaml
import argparse
from loguru import logger
from sklearn.preprocessing import MaxAbsScaler

from config import random_state
from config import data_dict_baseline as data_dict
from config import FEAT_BL_DIR as FEAT_DIR
from config import MODELS_BL_DIR as MODELS_DIR
from config import RESULTS_BL_DIR as RESULTS_DIR
from config import LOGS_BL_DIR as LOGS_DIR

from baselines.train import run_model
from utils.general_functs import create_dir, load_pickle


def load_data(args, data_dict):

    X_train = load_pickle(data_dict['train_feat_file'])
    X_test = load_pickle(data_dict['test_feat_file'])
    if os.path.isfile(data_dict['featname_file']):
        feature_names = load_pickle(data_dict['featname_file'])
    else:
        feature_names = None
    if os.path.isfile(data_dict['train_y_file']):
        y_train = load_pickle(data_dict['train_y_file'])
    else:
        y_train = None
    if os.path.isfile(data_dict['test_y_file']):
        y_test = load_pickle(data_dict['test_y_file'])
    else:
        y_test = None
    log_path = os.path.join(args.LOGS_DIR, '{}.log.txt'.format(args.cohort))

    return X_train, y_train, X_test, y_test, feature_names, log_path
    

def main(args):
    t0 = time.time()

    config = yaml.safe_load(open("baselines/train_config.yaml", 'r'))
    args.n_jobs = config['n_jobs']
    args.search_budget = config['search_budget']
    args.save_output = config['save_output']
    args.batch_size = config['batch_size']
    args.n_epochs = config['n_epochs']
    args.early_stopping_patience = config['early_stopping_patience']
    args.search_algo = config['search_algo']
    args.verbose_lvl = config['verbose_lvl']
    args.random_seed = random_state

    logger.info("Loading {} features", args.feat_type)
    X_train, y_train, X_test, y_test, feature_names, args.log_path = load_data(args, data_dict)

    logger.info("Scaling features")
    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Feature:", args.feat_type+" "+args.feat_transform, file=open(args.log_path, "a"))

    logger.info("Training and validating LR model")
    run_model(X_train, y_train, X_test, y_test, 'LR', args)
    logger.info("Training and validating NB model")
    run_model(X_train, y_train, X_test, y_test, 'NB', args)
    logger.info("Training and validating RF model")
    run_model(X_train, y_train, X_test, y_test, 'RF', args)

    logger.info("DONE.")
    print(f"Overall took {(time.time()-t0)/60} mins.", file=open(args.log_path, "a"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feat_type', help='Feature type (demo, dx, demo_dx, etc.)', default='demo_dx_med_proc_lab')
    parser.add_argument('-ft', '--feat_transform', help='Feature transformation (bow, bow_tb)', default='bow')
    parser.add_argument('-c', '--cohort', help='MDD or CAD cohort', default='MDD')
    args = parser.parse_args()

    if args.cohort.lower() == "mdd":
        logger.info("Loading MDD cohort")
        args.FEAT_DIR = FEAT_DIR
        args.MODELS_DIR = MODELS_DIR
        args.RESULTS_DIR = RESULTS_DIR
        args.LOGS_DIR = LOGS_DIR

    elif args.cohort.lower() == "cad":
        logger.info("Loading CAD cohort")

    else:
        raise ValueError("Wrong arguments: cohort")
    
    create_dir(args.MODELS_DIR) # create model directory if it doesn't exist
    create_dir(args.RESULTS_DIR) # create result directory if it doesn't exist
    create_dir(args.LOGS_DIR) # create log directory if it doesn't exist

    main(args)
