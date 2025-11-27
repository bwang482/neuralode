
import os
import re
import torch
import logging
import socket
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )


def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"
   output_path = os.path.join("/home/bw720/nde_traj/src/logs", file_prefix + ".pickle")

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(output_path)
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return
   

def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


def parse_model_filename(model_path: str) -> Dict[str, Any]:
    """
    Parse model filename to extract dataset info and hyperparameters.
    
    Expected filename format:
    final_latent_ode_<dataset>_<window>_<feature_type>_auc<auc>_lr<lr>_<hyperparams>_<date>_<time>.pt
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing parsed information
    """
    model_filename = os.path.basename(model_path)
    
    # Remove the .pt extension
    filename_without_ext = model_filename.rsplit('.', 1)[0]
    
    # Split by underscore
    parts = filename_without_ext.split('_')
    
    # Initialize result dictionary with defaults
    result = {
        # Model info
        'dataset_name': None,
        'lookback_window': None,
        'feature_type': None,
        'auc': None,
        'learning_rate': None,
        
        # Hyperparameters with defaults
        'n_traj': None,
        'ce_weight': None,
        'gl_dim': None,
        'rl_dim': None,
        'g_layer': None,
        'r_layer': None,
        'n_gru': None,
        'n_classif': None,
        'sample_tp': None,
        'z_last': None,
        
        # Metadata
        'date': None,
        'time': None
    }
    
    # Parse fixed position elements
    if len(parts) >= 6:
        # Skip "final_latent_ode" prefix (parts 0-2)
        result['dataset_name'] = parts[3]  # e.g., "CAD"
        result['lookback_window'] = parts[4]  # e.g., "1y" or "2y"
        
        # Feature type might be "demo-dx" or "demo-dx-med"
        # Check if the next part contains "demo"
        for i in range(5, len(parts)):
            if 'demo' in parts[i]:
                result['feature_type'] = parts[i]
                break
    
    # Parse hyperparameters and other values
    for part in parts:
        # AUC score
        if part.startswith('auc'):
            auc_str = part[3:]
            # Convert from format like "0p873" to 0.873
            result['auc'] = float(auc_str.replace('p', '.'))
        
        # Learning rate
        elif part.startswith('lr'):
            lr_str = part[2:]
            # Convert from format like "1p000e03" to scientific notation
            result['learning_rate'] = float(lr_str.replace('p', '.').replace('e', 'e-'))
        
        # Hyperparameters
        elif part.startswith('ntraj'):
            result['n_traj'] = int(part[5:])
        elif part.startswith('cew'):
            result['ce_weight'] = int(part[3:])
        elif part.startswith('zlast'):
            result['z_last'] = part[5:].lower() == 'true'
        elif part.startswith('gldim'):
            result['gl_dim'] = int(part[5:])
        elif part.startswith('rldim'):
            result['rl_dim'] = int(part[5:])
        elif part.startswith('glayer'):
            result['g_layer'] = int(part[6:])
        elif part.startswith('rlayer'):
            result['r_layer'] = int(part[6:])
        elif part.startswith('ngru'):
            result['n_gru'] = int(part[4:])
        elif part.startswith('nclassif'):
            result['n_classif'] = int(part[8:])
        elif part.startswith('sampletp'):
            tp_str = part[8:]
            if tp_str != "None":
                result['sample_tp'] = float(tp_str.replace('p', '.'))
        
        # Date (format: YYYYMMDD)
        elif re.match(r'^\d{8}$', part):
            result['date'] = part
        
        # Time (format: HHMM)
        elif re.match(r'^\d{4}$', part) and result['date'] is not None:
            result['time'] = part
    
    return result


def format_model_filename(parsed_info: Dict[str, Any]) -> str:
    filename_parts = {
        "Dataset": parsed_info['dataset_name'],
        "look_back": parsed_info['lookback_window'],
        "feature_type": parsed_info['feature_type'],
        "validation_auc": parsed_info['auc'],
        "lr": parsed_info['learning_rate'],
        'n_traj': parsed_info['n_traj'],
        'ce_weight': parsed_info['ce_weight'],
        'gen_latent_dims': parsed_info['gl_dim'],
        'rec_latent_dims': parsed_info['rl_dim'],
        'gen_layers': parsed_info['g_layer'],
        'rec_layers': parsed_info['r_layer'],
        'gru_units': parsed_info['n_gru'],
        'classif_units': parsed_info['n_classif'],
        'sample_tp': parsed_info['sample_tp'],
        'z_last': parsed_info['z_last']
    }
    return filename_parts



