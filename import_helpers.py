import sys
import numpy as np
import csv
import scipy.io

# Prompt for a variable with optional unit and default, and convert to input_type
def prompt(msg, unit=None, default=None, input_type=str, allow_empty=False):
  while True:
    # Build the suffix based on unit and default
    if unit is not None and default is not None:
      suffix = f" [{unit}, default: {default}]: "
    elif default is not None:
      suffix = f" [default: {default}]: "
    elif unit is not None:
      suffix = f" [{unit}]: "
    else:
      suffix = ": "
    
    val = input(f"{msg.rstrip()}{suffix}")
    
    if val == '' and default is not None:
      return default
    if val == '' and allow_empty:
      return ''
    try:
      return input_type(val)
    except Exception:
      print(f"Invalid input. Expected {input_type.__name__}.")

# Prompt for a yes/no boolean with default
def prompt_yes_no(msg, default=False):
  while True:
    val = input(f"{msg} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if val == '' and default is not None:
      return default
    if val in ('y', 'yes'): return True
    if val in ('n', 'no'): return False
    if val == '': return default
    print("Please enter y or n.")

### --- Import functions for different file types ---
# Import from .mat file
def import_mat(filename, has_time, sample_rate=None):
  # Open MAT file and list variables
  mat = scipy.io.loadmat(filename)
  var_names = [k for k in mat.keys() if not k.startswith('__')]
  print("Variables in .mat file:")
  for i, name in enumerate(var_names):
    arr = mat[name]
    print(f"  [{i}] {name}: shape {arr.shape}, dtype {arr.dtype}")
  
  # Select shim channel variable
  shim_var = None
  while shim_var is None:
    idx = prompt(f"Select variable for shim channels", default=None, input_type=str, allow_empty=False)
    if idx.isdigit() and 0 <= int(idx) < len(var_names):
      shim_var = var_names[int(idx)]
    else:
      print("Invalid selection.")
  data = np.array(mat[shim_var])

  # Get time axis variable if has_time is True
  if has_time:
    time_var = None
    while time_var is None:
      idx = prompt(f"Select variable for time points", default=None, input_type=str, allow_empty=False)
      if idx.isdigit() and 0 <= int(idx) < len(var_names):
        time_var = var_names[int(idx)]
      else:
        print("Invalid selection.")
    time = np.array(mat[time_var]).squeeze()
    if data.ndim == 1:
      data = data[:, np.newaxis]
    # Find which axis in data matches the time length
    match_axis = None
    for axis, dim in enumerate(data.shape):
      if dim == time.shape[0]:
        match_axis = axis
        break
    if match_axis is None:
      print(f"Error: None of the shim array axes match the time array length ({time.shape[0]}). Shim shape: {data.shape}, time shape: {time.shape}")
      sys.exit(1)
    # Move the matching axis to axis 0 (samples), the other to axis 1 (channels)
    if match_axis != 0:
      data = np.moveaxis(data, match_axis, 0)
    # Now data.shape[0] == time.shape[0]
  else:
    # Generate time vector from sample_rate
    if sample_rate is None:
      raise ValueError("sample_rate must be provided if no time vector")
    n_samples = data.shape[0]
    time = np.arange(n_samples) / (sample_rate * 1e3)  # seconds
  
  # Get channel count and validate
  channels = data.shape[1] if data.ndim > 1 else 1
  if channels < 1 or channels > 64:
    print(f"Error: Number of channels must be 1-64, got {channels}")
    sys.exit(1)
  n_bd = (channels + 7) // 8

  # Pad to multiple of 8 channels if needed
  if channels < n_bd * 8:
    print(f"Padding from {channels} to {n_bd * 8} channels with zeros")
    padding = np.zeros((data.shape[0], n_bd * 8 - channels))
    data = np.concatenate([data, padding], axis=1)
  
  return time, data

# Import from CSV file
def import_csv(filename, has_time, sample_rate=None):
  with open(filename, newline='') as f:
    first_line = f.readline().strip()
    f.seek(0)
    # Determine delimiter: check if comma or space is more common in first line
    if ',' in first_line:
      delimiter = ','
    elif ' ' in first_line or '\t' in first_line:
      delimiter = None  # whitespace
    else:
      delimiter = ','  # default to comma
    
    reader = csv.reader(f, delimiter=delimiter, skipinitialspace=True) if delimiter == ',' else \
             csv.reader((line.replace('\t', ' ') for line in f), delimiter=' ', skipinitialspace=True)
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
  if not rows:
    print("No data found in CSV.")
    sys.exit(1)
  arr = np.array([[float(x) for x in row] for row in rows])
  if has_time:
    time = arr[:, 0]
    data = arr[:, 1:]
  else:
    data = arr
    if sample_rate is None:
      raise ValueError("sample_rate must be provided if no time vector")
    n_samples = data.shape[0]
    time = np.arange(n_samples) / (sample_rate * 1e3)  # seconds
  channels = data.shape[1]
  if channels > 64:
    print(f"Error: Number of channels must be <= 64, got {channels}")
    sys.exit(1)
  n_bd = (channels + 7) // 8

  if channels < n_bd * 8:
    print(f"Padding from {channels} to {n_bd * 8} channels with zeros")
    padding = np.zeros((data.shape[0], n_bd * 8 - channels))
    data = np.concatenate([data, padding], axis=1)
  return time, data

# Import from .npy file
def import_npy(filename, has_time, sample_rate=None):
  arr = np.load(filename)
  if arr.ndim == 1:
    arr = arr[:, np.newaxis]
  if has_time:
    time = arr[:, 0]
    data = arr[:, 1:]
  else:
    data = arr
    if sample_rate is None:
      raise ValueError("sample_rate must be provided if no time vector")
    n_samples = data.shape[0]
    time = np.arange(n_samples) / (sample_rate * 1e3)  # seconds
  channels = data.shape[1]
  if channels < 1 or channels > 64:
    print(f"Error: Number of channels must be 1-64, got {channels}")
    sys.exit(1)
  n_bd = (channels + 7) // 8
  if channels < n_bd * 8:
    print(f"Padding from {channels} to {n_bd * 8} channels with zeros")
    padding = np.zeros((data.shape[0], n_bd * 8 - channels))
    data = np.concatenate([data, padding], axis=1)
  return time, data
