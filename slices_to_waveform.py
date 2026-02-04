#!/usr/bin/env python3
"""
Slices to Waveform Converter

Converts .npy, .csv, or .mat files containing slice data to ramped waveform .npy files.
The output .npy file includes a time vector and can be used as input to convert_waveform.py.

- For .mat: Prompts for variable names for channels.
- For .csv/.npy: Automatically detects format.
- Adds ramping between slice transitions with configurable ramp samples and time.
"""

# ============================================================================
# USER CONFIGURABLE DEFAULTS
# ============================================================================
DEFAULT_RAMP_SAMPLES = 1  # Number of samples in ramp (1 = no ramp)
DEFAULT_RAMP_TIME_MS = 0.2  # Time per ramp in milliseconds

# ============================================================================
import sys
import os
import numpy as np
import csv

try:
  import scipy.io
except ImportError:
  scipy = None

def prompt(msg, default=None, type_=str, allow_empty=False):
  """Prompt user for input with optional default value."""
  while True:
    val = input(msg)
    if val == '' and default is not None:
      return default
    if val == '' and allow_empty:
      return ''
    try:
      return type_(val)
    except Exception:
      print(f"Invalid input. Expected {type_.__name__}.")

def import_mat(filename):
  """Import data from .mat file."""
  if scipy is None:
    print("scipy.io is required for .mat files.")
    sys.exit(1)
  mat = scipy.io.loadmat(filename)
  var_names = [k for k in mat.keys() if not k.startswith('__')]
  print("Variables in .mat file:")
  for i, name in enumerate(var_names):
    arr = mat[name]
    print(f"  [{i}] {name}: shape {arr.shape}, dtype {arr.dtype}")
  
  # Select shim channel variable
  while True:
    idx = input(f"Select variable for shim channels (0-{len(var_names)-1}): ").strip()
    if idx.isdigit() and 0 <= int(idx) < len(var_names):
      shim_var = var_names[int(idx)]
      break
    print("Invalid selection.")
  
  data = np.array(mat[shim_var])
  
  # Ensure data is 2D: [n_samples, n_channels]
  if data.ndim == 1:
    data = data[:, np.newaxis]
  elif data.ndim > 2:
    print(f"Error: Expected 1D or 2D array, got {data.ndim}D array with shape {data.shape}")
    sys.exit(1)
  
  # If shape is [n_channels, n_samples], transpose
  if data.shape[0] < data.shape[1]:
    print(f"Transposing data from shape {data.shape} to {data.T.shape}")
    data = data.T
  
  return data

def import_csv(filename):
  """Import data from .csv file."""
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
    
    if delimiter == ',':
      reader = csv.reader(f, delimiter=delimiter, skipinitialspace=True)
    else:
      reader = csv.reader((line.replace('\t', ' ') for line in f), delimiter=' ', skipinitialspace=True)
    
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
  
  if not rows:
    print("No data found in CSV.")
    sys.exit(1)
  
  data = np.array([[float(x) for x in row] for row in rows])
  return data

def import_npy(filename):
  """Import data from .npy file."""
  data = np.load(filename)
  
  # Ensure data is 2D
  if data.ndim == 1:
    data = data[:, np.newaxis]
  elif data.ndim > 2:
    print(f"Error: Expected 1D or 2D array, got {data.ndim}D array with shape {data.shape}")
    sys.exit(1)
  
  return data

def add_ramping(data, ramp_samples, ramp_time_ms):
  """
  Add ramping between slice transitions.
  
  Args:
    data: numpy array of shape [n_slices, n_channels] containing slice currents
    ramp_samples: number of interpolation samples between slices (1 = no ramp)
    ramp_time_ms: time duration of ramp in milliseconds
  
  Returns:
    ramp_array: numpy array of shape [n_total_samples, n_channels + 1]
                where first column is time in seconds, rest are channel currents
  """
  n_slices, n_channels = data.shape
  
  if ramp_samples == 1:
    # No ramping: just add zero time column
    ramp_array = np.zeros((n_slices + 1, n_channels + 1))
    ramp_array[1:, 1:] = data
    return ramp_array[1:, :]
  
  # Create output array: time and channels
  n_total_samples = ramp_samples * n_slices + 1
  ramp_array = np.zeros((n_total_samples, n_channels + 1))
  
  # Set the main samples (every rampo_samples-th row)
  ramp_array[ramp_samples::ramp_samples, 1:] = data
  
  # Interpolate between samples
  for i in range(1, ramp_samples):
    # Linear interpolation weight
    weight = i / ramp_samples
    
    # Interpolate channel values
    prev_samples = ramp_array[0:-ramp_samples:ramp_samples, 1:]
    next_samples = ramp_array[ramp_samples::ramp_samples, 1:]
    ramp_array[i::ramp_samples, 1:] = weight * (next_samples - prev_samples) + prev_samples
    
    # Set time values (incremental within each ramp)
    ramp_array[(i+1)::ramp_samples, 0] = weight * ramp_time_ms * 1e-3
  
  # Return without the first row (which is all zeros)
  return ramp_array[1:, :]

def main():
  """Main function."""
  # Get filename
  if len(sys.argv) > 1:
    filename = sys.argv[1]
  else:
    filename = input("Path to input file (.npy/.csv/.mat): ").strip()
  
  if not os.path.exists(filename):
    print(f"File not found: {filename}")
    sys.exit(1)
  
  ext = os.path.splitext(filename)[1].lower()
  
  # Load data based on file type
  print(f"Loading data from {filename}...")
  if ext == '.mat':
    data = import_mat(filename)
  elif ext == '.csv':
    data = import_csv(filename)
  elif ext == '.npy':
    data = import_npy(filename)
  else:
    print(f"Unsupported file extension: {ext}")
    sys.exit(1)
  
  print(f"Loaded data shape: {data.shape} (slices x channels)")
  
  # Get ramping parameters
  ramp_samples = prompt(
    f"Number of samples in ramp (default {DEFAULT_RAMP_SAMPLES}, 1 = no ramp): ",
    default=DEFAULT_RAMP_SAMPLES,
    type_=int
  )
  
  if ramp_samples < 1:
    print("Error: Number of ramp samples must be >= 1")
    sys.exit(1)
  
  if ramp_samples > 1:
    ramp_time_ms = prompt(
      f"Ramp time in milliseconds (default {DEFAULT_RAMP_TIME_MS}): ",
      default=DEFAULT_RAMP_TIME_MS,
      type_=float
    )
    
    if ramp_time_ms <= 0:
      print("Error: Ramp time must be positive")
      sys.exit(1)
  else:
    ramp_time_ms = DEFAULT_RAMP_TIME_MS
  
  # Add ramping
  print(f"Adding ramping with {ramp_samples} samples and {ramp_time_ms} ms per ramp...")
  assert(ramp_samples >= 1)
  ramped_data = add_ramping(data, ramp_samples, ramp_time_ms)
  
  print(f"Output data shape: {ramped_data.shape} (samples x [time + channels])")
  
  # Determine output filename
  base_name = os.path.splitext(os.path.basename(filename))[0]
  
  if ramp_samples == 1:
    default_output = f"{base_name}.csv"
  else:
    default_output = f"{base_name}_{ramp_samples}_samp_{ramp_time_ms}_ms.csv"
  
  output_filename = input(f"Output filename (default: {default_output}): ").strip()
  if not output_filename:
    output_filename = default_output
  
  # Ensure .csv extension
  if not output_filename.endswith('.csv'):
    output_filename += '.csv'
  
  # Save output as space-separated CSV
  np.savetxt("csv_wfms/"+output_filename, ramped_data, delimiter=' ', fmt='%.6f')
  print(f"Saved ramped waveform to: csv_wfms/{output_filename}")
  print(f"\nYou can now use this file with convert_waveform.py:")
  print(f"  python convert_waveform.py {output_filename}")

if __name__ == "__main__":
  main()
