#!/usr/bin/env python3
"""
Unified DAC Waveform Converter

Converts .npy, .csv, or .mat files to DAC waveform files for the rev_d_shim project.
Handles optional time vector in the input file.

- For .mat: Prompts for variable names for channels and (optionally) time.
- For .csv/.npy: Prompts if the first column is a time vector.
"""

# ============================================================================
# USER CONFIGURABLE DEFAULTS
# ============================================================================
# --- Dump time+channelA as .npy for import_npy ---
DUMP_NPY = False  # Set True to enable dumping time+channelA as .npy
DUMP_NPY_FILENAME = None  # Set to None for auto-naming, or provide a string

# --- Prompt Defaults ---
DEFAULT_DO_ADC_READOUT = True  # Create ADC readout command file?
DEFAULT_ADC_SAMPLE_RATE_KSPS = 62.5  # Default ADC sample rate in ksps for prompt
DEFAULT_HAS_TIME_VECTOR = True  # Does the file include a time vector column?
DEFAULT_SPI_CLOCK_FREQ_MHZ = 20  # SPI clock frequency in MHz
DEFAULT_SAMPLE_RATE_KSPS = None  # Sample rate in ksps (None means prompt required)
DEFAULT_ENABLE_COMPRESSION = True  # Enable sample compression (skip duplicate samples)?
DEFAULT_CREATE_ZERO_WAVEFORM = False  # Create equivalent zeroed waveform?
DEFAULT_ZERO_AT_END = False  # Zero at the end?
DEFAULT_EXTRA_ZERO_TIME_MS = 0  # Time (ms) to wait before final zero sample
DEFAULT_ADC_EXTRA_TIME_MS = 0.5  # Extra sample time after DAC completes (ms)

# ============================================================================
import sys
import os
import numpy as np
import math
import csv
import scipy.io

MAX_DELAY = 2**25 - 1  # Maximum delay value supported by hardware

def prompt(msg, unit=None, default=None, type_=str, allow_empty=False):
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
      return type_(val)
    except Exception:
      print(f"Invalid input. Expected {type_.__name__}.")

def prompt_yes_no(msg, default=False):
  while True:
    val = input(f"{msg} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if val == '' and default is not None:
      return default
    if val in ('y', 'yes'): return True
    if val in ('n', 'no'): return False
    if val == '': return default
    print("Please enter y or n.")

def current_to_dac_value(current_amps):
  arr = np.asarray(current_amps)
  # arr shape: [n_bd, 8, n_samples] or [8, n_samples] or [n_samples]
  dac = np.round(arr * 32767 / 5.1).astype(int)
  dac = np.clip(dac, -32767, 32767)
  return dac

def calculate_sample_delay(sample_rate_ksps, spi_clock_freq_mhz):
  sample_rate_hz = sample_rate_ksps * 1000
  spi_clock_freq_hz = spi_clock_freq_mhz * 1e6
  cycles_per_sample = spi_clock_freq_hz / sample_rate_hz
  delay = int(cycles_per_sample)
  return max(1, min(MAX_DELAY, delay))

def create_zeroed_samples(sample_count, n_channels):
  return [[0]*n_channels for _ in range(sample_count)]

def trim_and_zero_channels(samples_A):
  """
  Interactively prompt user for channel range to keep, zero and trim others, and pad to multiple of 8.
  Returns (samples_A, ch_start, ch_end, n_channels, n_bd)
  """
  print(f"Imported data shape: {samples_A.shape}")
  total_channels = samples_A.shape[1] if samples_A.ndim > 1 else 1
  print(f"Total channels in data: {total_channels}")
  if total_channels > 1:
    while True:
      ch_range = prompt(f"Channel range to keep (e.g. 0-7)", default=f"0-{total_channels-1}", type_=str)
      if '-' in ch_range:
        parts = ch_range.split('-')
        try:
          ch_start = int(parts[0])
          ch_end = int(parts[1]) + 1
          if not (0 <= ch_start < ch_end <= total_channels):
            print(f"Invalid range. Must be within 0-{total_channels-1}.")
            continue
          break
        except Exception:
          print(f"Invalid range format. Please enter as start-end, e.g. 0-7.")
          continue
      else:
        print(f"Invalid range format. Please enter as start-end, e.g. 0-7.")
        continue
    # Zero and trim channels outside selected range
    if ch_start > 0:
      samples_A[:, :ch_start] = 0.0
    if ch_end < total_channels:
      samples_A[:, ch_end:] = 0.0
    samples_A = samples_A[:, ch_start:ch_end]
    print(f"Trimmed to channels {ch_start}-{ch_end-1}, new shape: {samples_A.shape}")
    n_channels = samples_A.shape[1]
    n_bd = (n_channels + 7) // 8
    if n_channels < n_bd * 8:
      print(f"Padding from {n_channels} to {n_bd * 8} channels with zeros")
      padding = np.zeros((samples_A.shape[0], n_bd * 8 - n_channels))
      samples_A = np.concatenate([samples_A, padding], axis=1)
      n_channels = samples_A.shape[1]
  else:
    ch_start, ch_end = 0, 1
    n_bd = 1
    n_channels = 1
  # Reshape to [n_bd, 8, n_samples]
  n_samples = samples_A.shape[0]
  bd_samples_A = samples_A.T.reshape(n_bd, 8, n_samples)
  print(f"Final array shape: {bd_samples_A.shape}")
  return bd_samples_A

def write_all_waveform_files(outname, time_cycles, bd_samples_DAC, spi_clock_freq, filename, enable_compression=True, is_zeroed=False):
  """Write waveform files for all boards with per-board compression."""
  n_bd = bd_samples_DAC.shape[0]
  n_samples = bd_samples_DAC.shape[2]
  max_delay = MAX_DELAY
  
  # Create file handles for all boards
  file_handles = []
  wfm_filenames = []
  
  try:
    for bd in range(n_bd):
      wfm_filename = f"{outname}_bd{bd}.wfm" if n_bd > 1 else (outname if outname.endswith('.wfm') else f"{outname}.wfm")
      wfm_filenames.append(wfm_filename)
      f = open(wfm_filename, 'w')
      file_handles.append(f)
      
      # Write headers
      waveform_type = "Zeroed DAC Waveform" if is_zeroed else "DAC Waveform"
      f.write(f"# {waveform_type} File\n")
      f.write(f"# Source file: {filename}\n")
      f.write(f"# SPI clock frequency: {spi_clock_freq:.6g} MHz\n")
      f.write(f"# Number of samples: {n_samples}\n")
      f.write(f"# Board: {wfm_filename}\n")
      f.write(f"# Channels: 8\n")
      f.write("# Format:\n")
      f.write("#   T <count> <ch0-ch7> (trigger)\n")
      f.write("#   NT <count> (noop trigger)\n")
      f.write("#   D <delay> <ch0-ch7> (delay)\n")
      f.write("#   ND <delay> (noop delay)\n")
    
    # Process each board independently
    for bd in range(n_bd):
      last_written_time = None
      last_written_vals = None
      
      if enable_compression:
        # Pending trigger tracking
        pending_trigger_count = 0
        pending_trigger_vals = None
        
        # Pending delay tracking (for when values stay the same)
        time_pending = None  # Time when we started pending delays
        pending_vals = None   # Values for pending delay
        current_vals = None
        
        def flush_pending_delays():
          """Flush accumulated delays as ND command."""
          nonlocal time_pending, last_written_time
          if time_pending is not None and last_written_time is not None:
            delay = time_pending - last_written_time
            if delay > 0:
              delay_left = delay
              while delay_left > max_delay:
                file_handles[bd].write(f"ND {max_delay - 1000}\n")
                delay_left -= (max_delay - 1000)
              file_handles[bd].write(f"ND {delay_left}\n")
              last_written_time = time_pending
            time_pending = None
        
        def flush_pending_triggers():
          """Flush accumulated triggers as NT [count] command."""
          nonlocal pending_trigger_count, pending_trigger_vals
          if pending_trigger_count > 1:
            # Write the accumulated repeats (count-1 because first was already written)
            file_handles[bd].write(f"NT {pending_trigger_count - 1}\n")
            pending_trigger_count = 0
            pending_trigger_vals = None
        
        for i in range(n_samples):
          t = time_cycles[i]
          vals = bd_samples_DAC[bd, :, i]
          
          # Check if this is a trigger (first sample or time == 0)
          is_trigger = (i == 0 or t == 0)
          
          # Check if values changed
          vals_changed = (current_vals is None or not np.array_equal(vals, current_vals))
          
          # Determine if we need to process this sample
          is_last = (i == n_samples - 1)
          next_is_trigger = (i < n_samples - 1 and time_cycles[i + 1] == 0)
          should_process = is_trigger or vals_changed or is_last or next_is_trigger
          
          if not should_process:
            # Values same as current, mark as pending if not already
            if time_pending is None:
              time_pending = t
              pending_vals = vals.copy()
            continue
          
          if is_trigger:
            # Flush any pending delays first
            flush_pending_delays()
            
            # Check if this trigger has same values as pending trigger
            if pending_trigger_count > 0 and np.array_equal(vals, pending_trigger_vals):
              # Same trigger as pending, increment count
              pending_trigger_count += 1
            else:
              # Different trigger or first trigger
              # Flush any existing pending triggers (accumulated repeats)
              if pending_trigger_count > 1:
                # Write the accumulated repeats (count-1 because first was already written)
                file_handles[bd].write(f"NT {pending_trigger_count - 1}\n")
              
              # Write this trigger immediately as T 1
              file_handles[bd].write(f"T 1" + ''.join(f" {v}" for v in vals) + "\n")
              last_written_time = 0
              last_written_vals = vals.copy()
              
              # Start tracking for potential repeats
              pending_trigger_count = 1
              pending_trigger_vals = vals.copy()
            
            current_vals = vals.copy()
            
          else:
            # Non-trigger command
            if last_written_time is None:
              raise ValueError("First sample must be a trigger")
            
            if vals_changed:
              # Values changed - flush pending triggers and delays, then write D command
              flush_pending_triggers()
              flush_pending_delays()
              
              # Calculate delay from last written time to this sample
              delay = t - last_written_time
              
              # Write D command with this delay and new values
              delay_left = delay
              while delay_left > max_delay:
                file_handles[bd].write(f"ND {max_delay - 1000}\n")
                delay_left -= (max_delay - 1000)
              
              file_handles[bd].write(f"D {delay_left}" + ''.join(f" {v}" for v in vals) + "\n")
              last_written_time = t
              last_written_vals = vals.copy()
              current_vals = vals.copy()
              
            else:
              # Values same as before, mark as pending
              if time_pending is None:
                time_pending = t
                pending_vals = vals.copy()
        
        # Flush any remaining pending commands at end
        flush_pending_delays()
        flush_pending_triggers()
        
      else:
        # No compression - write every command immediately
        for i in range(n_samples):
          t = time_cycles[i]
          vals = bd_samples_DAC[bd, :, i]
          
          # Check if this is a trigger (first sample or time == 0)
          is_trigger = (i == 0 or t == 0)
          
          if is_trigger:
            # Write trigger command
            file_handles[bd].write(f"T 1" + ''.join(f" {v}" for v in vals) + "\n")
            last_written_time = 0
            last_written_vals = vals.copy()
          else:
            # Non-trigger command
            if last_written_time is None:
              raise ValueError("First sample must be a trigger")
            
            # Calculate delay from last written time to this sample
            delay = t - last_written_time
            
            # Write D command with this delay and values
            delay_left = delay
            while delay_left > max_delay:
              file_handles[bd].write(f"ND {max_delay - 1000}\n")
              delay_left -= (max_delay - 1000)
            
            file_handles[bd].write(f"D {delay_left}" + ''.join(f" {v}" for v in vals) + "\n")
            last_written_time = t
            last_written_vals = vals.copy()
    
    # Close all files and print success messages
    for bd in range(n_bd):
      file_handles[bd].close()
      print(f"Waveform file written to: {wfm_filenames[bd]}")
    
  except IOError as e:
    # Close any open files on error
    for f in file_handles:
      if not f.closed:
        f.close()
    print(f"Error writing waveform files: {e}")
    sys.exit(1)
  except Exception as e:
    # Close any open files on error
    for f in file_handles:
      if not f.closed:
        f.close()
    raise e

def calculate_dac_durations(time):
  """
  Given a time vector in clock cycles, return a list of durations (in clock cycles) for each trigger segment.
  Each duration is the last non-zero time before the next time==0 (or end).
  """
  time = np.asarray(time)
  zero_idxs = np.where(time == 0)[0]
  durations = []
  for i, idx in enumerate(zero_idxs):
    if i + 1 < len(zero_idxs):
      next_idx = zero_idxs[i+1]
      seg = time[idx:next_idx]
    else:
      seg = time[idx:]
    # Find last nonzero time in this segment
    nonzero = seg[seg > 0]
    if len(nonzero) > 0:
      duration = int(nonzero[-1])
    else:
      duration = 0
    durations.append(duration)
  return durations

def write_adc_readout_file(filename, durations_cycles, adc_sample_rate_ksps, extra_time_ms, spi_clock_freq_mhz):
  try:
    with open(filename, 'w') as f:
      # Convert extra_cycles back to true extra_time_ms (in case of rounding)
      extra_cycles = int(round(extra_time_ms * 1e-3 * spi_clock_freq_mhz * 1e6))
      true_extra_time_ms = extra_cycles / (spi_clock_freq_mhz * 1e6) * 1e3
      # Convert durations_cycles to ms for comment
      durations_ms = [cycles / (spi_clock_freq_mhz * 1e6) * 1e3 for cycles in durations_cycles]
      f.write("# ADC Readout Command File\n")
      f.write(f"# Extra sample time: {true_extra_time_ms:.6g} ms\n")
      f.write(f"# SPI clock frequency: {spi_clock_freq_mhz:.6g} MHz\n")
      f.write(f"# ADC sample rate: {adc_sample_rate_ksps:.6g} ksps\n")
      # f.write("O 0 1 2 3 4 5 6 7\n")
      for i, dur_cycles in enumerate(durations_cycles):
        total_cycles = dur_cycles + extra_cycles
        adc_delay_value = calculate_sample_delay(adc_sample_rate_ksps, spi_clock_freq_mhz)
        # Estimate total samples as total_cycles // adc_delay_value
        total_samples = max(1, total_cycles // adc_delay_value)
        f.write("NT 1\n")
        repeat_count = total_samples - 1
        f.write(f"D {adc_delay_value} {repeat_count}\n")
    print(f"ADC readout file written to: {filename}")
  except IOError as e:
    print(f"Error writing ADC readout file {filename}: {e}")
    sys.exit(1)

def get_adc_readout_parameters(dac_params):
  print("\n--- ADC Readout Parameters ---")
  default_adc_rate = dac_params['sample_rate'] if dac_params['sample_rate'] is not None else DEFAULT_ADC_SAMPLE_RATE_KSPS
  while True:
    adc_sample_rate = prompt("ADC sample rate", unit="ksps", default=default_adc_rate, type_=float)
    if adc_sample_rate > 0:
      break
    print("ADC sample rate must be positive")
  while True:
    extra_time = prompt("Extra sample time after DAC completes", unit="ms", default=DEFAULT_ADC_EXTRA_TIME_MS, type_=float)
    if extra_time >= 0:
      break
    print("Extra sample time must be non-negative")
  return {
    'adc_sample_rate': adc_sample_rate,
    'adc_extra_time': extra_time
  }


def import_mat(filename, has_time, sample_rate=None):
  mat = scipy.io.loadmat(filename)
  var_names = [k for k in mat.keys() if not k.startswith('__')]
  print("Variables in .mat file:")
  for i, name in enumerate(var_names):
    arr = mat[name]
    print(f"  [{i}] {name}: shape {arr.shape}, dtype {arr.dtype}")
  # Select shim channel variable
  shim_var = None
  while shim_var is None:
    idx = prompt(f"Select variable for shim channels", default=None, type_=str, allow_empty=False)
    if idx.isdigit() and 0 <= int(idx) < len(var_names):
      shim_var = var_names[int(idx)]
    else:
      print("Invalid selection.")
  data = np.array(mat[shim_var])
  if has_time:
    time_var = None
    while time_var is None:
      idx = prompt(f"Select variable for time points", default=None, type_=str, allow_empty=False)
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
  channels = data.shape[1] if data.ndim > 1 else 1
  if channels < 1 or channels > 64:
    print(f"Error: Number of channels must be 1-64, got {channels}")
    sys.exit(1)
  n_bd = (channels + 7) // 8
  if channels < n_bd * 8:
    print(f"Padding from {channels} to {n_bd * 8} channels with zeros")
    padding = np.zeros((data.shape[0], n_bd * 8 - channels))
    data = np.concatenate([data, padding], axis=1)
  n_channels = data.shape[1]
  return time, data

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
  n_bd = math.ceil(channels / 8)
  if channels < n_bd * 8:
    print(f"Padding from {channels} to {n_bd * 8} channels with zeros")
    padding = np.zeros((data.shape[0], n_bd * 8 - channels))
    data = np.concatenate([data, padding], axis=1)
  n_channels = data.shape[1]
  return time, data

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
  n_channels = data.shape[1]
  return time, data

def main():
  # Get filename
  if len(sys.argv) > 1:
    filename = sys.argv[1]
  else:
    filename = input("Path to input file (.npy/.csv/.mat): ").strip()
  if not os.path.exists(filename):
    print(f"File not found: {filename}")
    sys.exit(1)
  ext = os.path.splitext(filename)[1].lower()

  # Ask about ADC readout first
  do_adc_readout = prompt_yes_no("Do ADC readout?", default=DEFAULT_DO_ADC_READOUT)

  # Ask if there is a time vector
  has_time = prompt_yes_no("Does the file include a time vector column?", default=DEFAULT_HAS_TIME_VECTOR)

  # Get SPI clock and sample rate
  spi_clock_freq = prompt("SPI clock frequency", unit="MHz", default=DEFAULT_SPI_CLOCK_FREQ_MHZ, type_=float)
  if not has_time:
    sample_rate = prompt(f"Sample rate", unit="ksps", default=DEFAULT_SAMPLE_RATE_KSPS, type_=float)
  else:
    sample_rate = None
  
  # Get additional options
  enable_compression = prompt_yes_no("Enable sample compression (skip duplicate samples)?", default=DEFAULT_ENABLE_COMPRESSION)
  create_zero_waveform = prompt_yes_no("Create equivalent zeroed waveform?", default=DEFAULT_CREATE_ZERO_WAVEFORM)
  zero_at_end = prompt_yes_no("Zero at the end?", default=DEFAULT_ZERO_AT_END)
  params = {
    'sample_rate': sample_rate,
    'spi_clock_freq': spi_clock_freq,
    'enable_compression': enable_compression,
    'do_adc_readout': do_adc_readout,
    'create_zero_waveform': create_zero_waveform,
    'zero_at_end': zero_at_end
  }
  if do_adc_readout:
    adc_params = get_adc_readout_parameters(params)
    params.update(adc_params)
  
  # Load data and process by filetype
  if ext == '.mat':
    time, samples_A = import_mat(filename, has_time, sample_rate=sample_rate)
  elif ext == '.csv':
    time, samples_A = import_csv(filename, has_time, sample_rate=sample_rate)
  elif ext == '.npy':
    time, samples_A = import_npy(filename, has_time, sample_rate=sample_rate)
  else:
    print(f"Unsupported file extension: {ext}")
    sys.exit(1)
  

  # If zero_at_end, append a final zero sample if not already present
  if zero_at_end:
    # Check if last sample is already all zeros
    if not np.all(samples_A[-1] == 0):
      zero_row = np.zeros((1, samples_A.shape[1]))
      samples_A = np.vstack([samples_A, zero_row])
      # Prompt for time (in ms) to wait before the final zero
      extra_zero_time_ms = prompt("Time (ms) to wait before final zero sample", unit="ms", default=DEFAULT_EXTRA_ZERO_TIME_MS, type_=float)
      while extra_zero_time_ms < 0:
        print("Time must be non-negative")
        extra_zero_time_ms = prompt("Time (ms) to wait before final zero sample", unit="ms", default=DEFAULT_EXTRA_ZERO_TIME_MS, type_=float)
      # Set the final time entry
      time = np.append(time, time[-1] + extra_zero_time_ms / 1e3)

  # --- Dump time+channelA as .npy if enabled ---
  if DUMP_NPY:
    # Compose array: first column is time, rest are channel A values
    arr = np.column_stack((time, samples_A))
    if DUMP_NPY_FILENAME:
      npy_out = DUMP_NPY_FILENAME
    else:
      base = os.path.splitext(os.path.basename(filename))[0]
      npy_out = f"{base}_dump.npy"
    np.save(npy_out, arr)
    print(f"[DUMP_NPY] Saved time+channelA as: {npy_out}")

  # Convert and trim channels and time
  bd_samples_A = trim_and_zero_channels(samples_A)
  bd_samples_DAC = current_to_dac_value(bd_samples_A)

  # Convert time vector to integer clock cycles after trimming
  spi_clock_freq = params['spi_clock_freq']
  time_cycles = np.round(time * spi_clock_freq * 1e6).astype(int)
  # Ensure time_cycles is monotonic and starts at zero
  time_cycles = time_cycles - time_cycles[0]

  # Get output filename
  if bd_samples_DAC.size == 0:
    print("No samples to write")
    sys.exit(1)
  default_filename = os.path.splitext(os.path.basename(filename))[0]
  if params['sample_rate'] is not None:
    default_filename = f"{default_filename}_{params['sample_rate']:.0f}ksps"
  outname = input(f"Output filename (default: {default_filename}.[wfm/rdout]): ").strip()
  if not outname:
    outname = default_filename

  # Write waveform files
  write_all_waveform_files("waveforms/"+outname, time_cycles, bd_samples_DAC, params['spi_clock_freq'], filename, params['enable_compression'])

  if params.get('create_zero_waveform', False):
    # Make zero waveform: one trigger and one D per duration, D delay = duration
    zero_samples = np.zeros_like(bd_samples_DAC[0])
    zero_filename = f"waveforms/{outname}_zero.wfm"
    durations_cycles = calculate_dac_durations(time_cycles)
    try:
      with open(zero_filename, 'w') as f:
        f.write("# Zeroed DAC Waveform File (trigger and D per duration)\n")
        f.write(f"# Source file: {filename}\n")
        f.write(f"# SPI clock frequency: {params['spi_clock_freq']:.6g} MHz\n")
        f.write(f"# Board: {zero_filename}\n")
        f.write(f"# Channels: 8\n")
        f.write("# Format: T <count> <ch0-ch7> (trigger) / D <delay> <ch0-ch7> (delay)\n")
        for dur in durations_cycles:
          f.write("T 1" + ''.join(f" 0" for _ in range(zero_samples.shape[0])) + "\n")
          f.write(f"D {dur}" + ''.join(f" 0" for _ in range(zero_samples.shape[0])) + "\n")
      print(f"Zeroed waveform file written to: {zero_filename}")
    except IOError as e:
      print(f"Error writing zeroed waveform file {zero_filename}: {e}")
      sys.exit(1)
  
  # Always write ADC file
  durations_cycles = calculate_dac_durations(time_cycles)
  rdout_filename = outname if outname.endswith('.rdout') else f"{outname}.rdout"
  
  if params.get('do_adc_readout', False):
    # Write full ADC readout file
    write_adc_readout_file(
      "waveforms/"+rdout_filename,
      durations_cycles,
      params['adc_sample_rate'],
      params['adc_extra_time'],
      params['spi_clock_freq']
    )
  else:
    # Write minimal ADC file with only NT command
    n_triggers = len(durations_cycles)
    try:
      with open("waveforms/"+rdout_filename, 'w') as f:
        f.write("# ADC Readout Command File (No readout)\n")
        f.write(f"# Number of triggers: {n_triggers}\n")
        f.write(f"NT {n_triggers}\n")
      print(f"ADC readout file written to: waveforms/{rdout_filename}")
    except IOError as e:
      print(f"Error writing ADC readout file: {e}")
      sys.exit(1)
  print("Waveform generation complete!")

if __name__ == "__main__":
  main()
