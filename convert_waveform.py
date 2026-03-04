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
DEFAULT_SPI_CLK_FREQ_MHZ = 20  # SPI clock frequency in MHz
DEFAULT_SAMPLE_RATE_KSPS = None  # Sample rate in ksps (None means prompt required)
DEFAULT_ENABLE_COMPRESSION = True  # Enable sample compression (skip duplicate samples)?
DEFAULT_CREATE_ZERO_WAVEFORM = False  # Create equivalent zeroed waveform?
DEFAULT_ADC_EXTRA_TIME_MS = 0.5  # Extra sample time after DAC completes (ms)

# ============================================================================
import sys
import os
import numpy as np
from import_helpers import *

MAX_DELAY = 2**25 - 1  # Maximum delay value supported by hardware

# Convert current in amps to DAC values, from +/-5A to a signed 16-bit integer range
def current_to_dac_value(current_amps):
  arr = np.asarray(current_amps)
  # arr shape: [n_bd, 8, n_samples] or [8, n_samples] or [n_samples]
  dac = np.round(arr * 32767 / 5.0).astype(int)
  dac = np.clip(dac, -32767, 32767)
  return dac

# Calculate clock delay from sample rate
def calculate_sample_delay(sample_rate_ksps, spi_clk_freq_mhz):
  sample_rate_hz = sample_rate_ksps * 1000
  spi_clk_freq_hz = spi_clk_freq_mhz * 1e6
  cycles_per_sample = spi_clk_freq_hz / sample_rate_hz
  delay = int(cycles_per_sample)
  return max(1, min(MAX_DELAY, delay))

# Given samples_A of shape [n_samples, n_channels], prompt user for channel range to keep.
# Zero out channels below the selected range start, trim unused channels above the selected range end,
# and pad to multiple of 8 channels if needed.
def trim_channels(samples_A):
  """
  Interactively prompt user for channel range to keep, zero lower channels, trim unused upper channels,
  and pad to multiple of 8.
  Returns (samples_A, ch_start, ch_end, n_channels, n_bd)
  """
  print(f"Imported data shape: {samples_A.shape}")
  total_channels = samples_A.shape[1] if samples_A.ndim > 1 else 1
  print(f"Total channels in data: {total_channels}")
  if total_channels > 1:
    while True:
      ch_range = prompt(f"Channel range to keep (e.g. 0-7)", default=f"0-{total_channels-1}", input_type=str)
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
    # Zero lower-side channels outside selected range
    if ch_start > 0:
      samples_A[:, :ch_start] = 0.0
    # Trim unused upper-side channels outside selected range
    if ch_end < total_channels:
      samples_A = samples_A[:, :ch_end]
    n_channels = samples_A.shape[1]
    n_bd = (n_channels + 7) // 8
    if n_channels < n_bd * 8:
      print(f"Padding from {n_channels} to {n_bd * 8} channels with zeros")
      padding = np.zeros((samples_A.shape[0], n_bd * 8 - n_channels))
      samples_A = np.concatenate([samples_A, padding], axis=1)
      n_channels = samples_A.shape[1]
    print(f"Zeroed channels 0-{ch_start-1} and trimmed above {ch_end-1}.")
    print(f"Padded to fill out used boards. New shape: {samples_A.shape}")
  else:
    ch_start, ch_end = 0, 1
    n_bd = 1
    n_channels = 1
  # Reshape to [n_bd, 8, n_samples]
  n_samples = samples_A.shape[0]
  bd_samples_A = samples_A.T.reshape(n_bd, 8, n_samples)
  print(f"Final array shape: {bd_samples_A.shape}")
  return bd_samples_A

# Calculate the durations of each DAC trigger segment in clock cycles
# Each segment starts with a time==0 (trigger) and ends at the last non-zero time before the next time==0 or end of array.
def calculate_dac_durations(time):
  """
  Given a time vector in clock cycles, return a list of durations (in clock cycles) for each trigger segment.
  Each duration is the last non-zero time before the next time==0 (or end).
  """
  time = np.asarray(time)
  assert(time.ndim == 1)
  if time[0] != 0:
    raise ValueError("First sample must have time == 0 (trigger)")
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

# Segment time/channel data into boards and trigger segments
def segment_waveform(time_cycles, bd_samples_DAC):
  """
  Given [samples] time vector and [boards, 8, samples] data array, segment into boards and then into trigger segments.
  Returns:
  - boards: list of lists of trigger segments.
    - Each trigger segment is a dict with:
      - 'trigger_values': [8] array of values at trigger
      - 'delays': [segment_samples - 1] array of delays between subsequent samples
      - 'values': [segment_samples - 1, 8] array of values for subsequent samples
      - 'duration': total duration of the segment in clock cycles
  - min_delay: minimum delay between samples across all segments
  """
  n_bd = bd_samples_DAC.shape[0]
  n_samples = bd_samples_DAC.shape[2]
  if time_cycles.shape[0] != n_samples:
    raise ValueError(f"Time vector length {time_cycles.shape[0]} does not match number of samples {n_samples}")
  if time_cycles[0] != 0:
    raise ValueError("First sample must have time == 0 (trigger)")
  zero_idxs = np.where(time_cycles == 0)[0]
  boards = []
  min_delay = None
  for bd in range(n_bd):
    bd_segments = []
    for i, idx in enumerate(zero_idxs):
      next_idx = zero_idxs[i+1] if i + 1 < len(zero_idxs) else n_samples
      duration = time_cycles[next_idx - 1]
      trigger_values = bd_samples_DAC[bd, :, idx]
      if next_idx - idx > 1:
        delays = np.diff(time_cycles[idx:next_idx])
        values = bd_samples_DAC[bd, :, idx+1:next_idx].T
      else:
        delays = np.array([])
        values = np.array([]).reshape(0, 8)
      if len(delays) > 0:
        segment_min_delay = np.min(delays)
        if min_delay is None or segment_min_delay < min_delay:
          min_delay = segment_min_delay
      segment = {
        'trigger_values': trigger_values,
        'delays': delays,
        'values': values,
        'duration': duration
      }
      bd_segments.append(segment)
    boards.append(bd_segments)
  return boards, min_delay

# Compress segments
def compress_segment(segment):
  """
  Given a segment dict with 'trigger_values', 'delays', 'values', and 'duration',
  return a compressed segment dict where subsequent samples with the same values are removed.
  If there is a new value before the end of the segment, add the accumulated delay to its delay.
  """
  trigger_values = segment['trigger_values']
  delays = segment['delays']
  values = segment['values']

  if len(delays) == 0:
    return segment
  
  compressed_delays = []
  compressed_values = []
  prev_values = trigger_values
  accumulated_delay = 0
  for delay, val in zip(delays, values):
    if np.array_equal(val, prev_values):
      accumulated_delay += delay
    else:
      compressed_delays.append(accumulated_delay + delay)
      compressed_values.append(val)
      prev_values = val
      accumulated_delay = 0
  
  compressed_segment = {
    'trigger_values': trigger_values,
    'delays': np.array(compressed_delays),
    'values': np.array(compressed_values),
    'duration': segment['duration']
  }
  return compressed_segment

# Write a waveform file for a single board
def write_waveform_file(filename, segments, spi_clk_freq_mhz, min_delay, compress=True):
  """
  Write a waveform file for a single board given its trigger segments.
  Each segment has a trigger with values, followed by optional delays and value changes.
  If compress=True, apply sample compression by skipping samples with same values and accumulating delays/triggers
  """
  max_delay = MAX_DELAY
  try:
    with open(filename, 'w') as f:
      f.write(f"# DAC Waveform Command File\n")
      f.write(f"# Compiled for SPI clock frequency: {spi_clk_freq_mhz:.6g} MHz\n")
      f.write(f"# Command format:\n")
      f.write(f"#   T <count> <ch0-ch7> (Update to given values after [count] triggers)\n")
      f.write(f"#   NT <count> (Do nothing for [count] triggers)\n")
      f.write(f"#   D <delay> <ch0-ch7> (Update to given values after a delay of [delay] clock cycles)\n")
      f.write(f"#   ND <delay> (Do nothing [delay] clock cycles)\n")
      prev_values = None
      trigger_count = 0

      for segment in segments:
        if compress:
          segment = compress_segment(segment)
        
        trigger_values = segment['trigger_values']
        remaining_duration = segment['duration']

        skip_trigger = compress and prev_values is not None and np.array_equal(trigger_values, prev_values)
        
        # Accumulate skipped triggers if the same trigger values repeat across segments
        if (skip_trigger):
          trigger_count += 1
        
        # Emit accumulated skipped triggers if either there is a new trigger value or different delay value
        if (not skip_trigger or len(segment['delays']) > 0):
          # Write any accumulated triggers for previous values
          if trigger_count > 0:
            f.write(f"NT {trigger_count}\n")
            trigger_count = 0

        if not skip_trigger:
          # Write the trigger command for this segment
          f.write(f"T 1" + ''.join(f" {v}" for v in segment['trigger_values']) + "\n")
          prev_values = segment['trigger_values']

        for delay, values in zip(segment['delays'], segment['values']):
          # If compressing, we've already combined consecutive samples with same values
          # However, there's a maximum and minimum delay, so we may need to split
          # the delay into multiple commands if it exceeds max_delay (without going below min_delay)
          if compress:
            while delay > max_delay:
              if delay - max_delay < min_delay:
                # Emit a no-op to bright the delay to the min_delay threshold
                f.write(f"ND {delay - min_delay}\n")
                remaining_duration -= (delay - min_delay)
                delay = min_delay
              else:
                f.write(f"ND {max_delay}\n")
                remaining_duration -= max_delay
                delay -= max_delay
          f.write(f"D {delay}" + ''.join(f" {v}" for v in values) + "\n")
          remaining_duration -= delay
        
        # If there is remaining duration after the last sample, emit no-op delays to prevent any SPI commands until all other channels are done
        if remaining_duration > 0:
          while remaining_duration > max_delay:
            if remaining_duration - max_delay < min_delay:
              f.write(f"ND {remaining_duration - min_delay}\n")
              remaining_duration = min_delay
            else:
              f.write(f"ND {max_delay}\n")
              remaining_duration -= max_delay
          if remaining_duration > 0:
            f.write(f"ND {remaining_duration}\n")
        
  except IOError as e:
    print(f"Error writing waveform file {filename}: {e}")
    sys.exit(1)
      
def write_adc_readout_file(filename, durations_cycles, adc_sample_rate_ksps, extra_time_ms, spi_clk_freq_mhz):
  try:
    with open(filename, 'w') as f:
      # Convert extra_cycles back to true extra_time_ms (in case of rounding)
      extra_cycles = int(round(extra_time_ms * 1e-3 * spi_clk_freq_mhz * 1e6))
      true_extra_time_ms = extra_cycles / (spi_clk_freq_mhz * 1e6) * 1e3
      f.write("# ADC Readout Command File\n")
      f.write(f"# Compiled for SPI clock frequency: {spi_clk_freq_mhz:.6g} MHz\n")
      f.write(f"# Extra sample time: {true_extra_time_ms:.6g} ms\n")
      f.write(f"# ADC sample rate: {adc_sample_rate_ksps:.6g} ksps\n")
      f.write(f"# Command format:\n")
      f.write(f"#   NT <count> (Do nothing for [count] triggers)\n")
      f.write(f"#   D <delay> <repeat_count> (Sample and then wait until [delay] clock cycles has passed)\n")
      f.write(f"#                            (Repeat this sample+delay command [repeat_count] times)\n")
      for dur_cycles in durations_cycles:
        total_cycles = dur_cycles + extra_cycles
        adc_delay_value = calculate_sample_delay(adc_sample_rate_ksps, spi_clk_freq_mhz)
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
    adc_sample_rate = prompt("ADC sample rate", unit="ksps", default=default_adc_rate, input_type=float)
    if adc_sample_rate > 0:
      break
    print("ADC sample rate must be positive")
  while True:
    extra_time = prompt("Extra sample time after DAC completes", unit="ms", default=DEFAULT_ADC_EXTRA_TIME_MS, input_type=float)
    if extra_time >= 0:
      break
    print("Extra sample time must be non-negative")
  return {
    'adc_sample_rate': adc_sample_rate,
    'adc_extra_time': extra_time
  }

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
  spi_clk_freq_mhz = prompt("SPI clock frequency", unit="MHz", default=DEFAULT_SPI_CLK_FREQ_MHZ, input_type=float)
  if not has_time:
    sample_rate = prompt(f"Sample rate", unit="ksps", default=DEFAULT_SAMPLE_RATE_KSPS, input_type=float)
  else:
    sample_rate = None
  
  # Get additional options
  enable_compression = prompt_yes_no("Enable sample compression (skip duplicate samples)?", default=DEFAULT_ENABLE_COMPRESSION)
  create_zero_waveform = prompt_yes_no("Create equivalent zeroed waveform?", default=DEFAULT_CREATE_ZERO_WAVEFORM)
  params = {
    'sample_rate': sample_rate,
    'spi_clk_freq_mhz': spi_clk_freq_mhz,
    'enable_compression': enable_compression,
    'do_adc_readout': do_adc_readout,
    'create_zero_waveform': create_zero_waveform
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
  bd_samples_A = trim_channels(samples_A)
  bd_samples_DAC = current_to_dac_value(bd_samples_A)

  # Convert time vector to integer clock cycles after trimming
  spi_clk_freq_mhz = params['spi_clk_freq_mhz']
  time_cycles = np.round(time * spi_clk_freq_mhz * 1e6).astype(int)
  # Ensure time_cycles is monotonic and starts at zero
  time_cycles = time_cycles - time_cycles[0]

  # Segment into boards and trigger segments
  boards, min_delay = segment_waveform(time_cycles, bd_samples_DAC)

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
  for bd, segments in enumerate(boards):
    assert(len(segments) > 0)
    filename = f"waveforms/{outname}_bd{bd}.wfm"
    write_waveform_file(filename, segments, spi_clk_freq_mhz=spi_clk_freq_mhz, min_delay=min_delay, compress=enable_compression)
    print(f"Waveform file for board {bd} written to: {filename}")

  if params.get('create_zero_waveform', False):
    # Make zero waveform: one trigger and one D per duration, D delay = duration
    zero_filename = f"waveforms/{outname}_zero.wfm"
    zero_segments = []
    for segment in boards[0]:  # Just use the first board's segments for timing
      zero_segment = {
        'trigger_values': np.zeros(8, dtype=int),
        'delays': np.array([]),
        'values': np.array([]).reshape(0, 8),
        'duration': segment['duration']
      }
      zero_segments.append(zero_segment)
    write_waveform_file(zero_filename, zero_segments, spi_clk_freq_mhz=spi_clk_freq_mhz, min_delay=min_delay, compress=enable_compression)
    print(f"Zeroed waveform file written to: {zero_filename}")
  
  # Always write an ADC readout file -- "Do ADC readout?" just controls whether the file includes samples
  durations_cycles = calculate_dac_durations(time_cycles)
  rdout_filename = outname if outname.endswith('.rdout') else f"{outname}.rdout"
  
  if params.get('do_adc_readout', False):
    # Write full ADC readout file
    write_adc_readout_file(
      "waveforms/"+rdout_filename,
      durations_cycles,
      params['adc_sample_rate'],
      params['adc_extra_time'],
      params['spi_clk_freq_mhz']
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
