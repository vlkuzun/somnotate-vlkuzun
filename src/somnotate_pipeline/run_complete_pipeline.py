#!/usr/bin/env python
# filepath: /Users/Volkan/Repos/somnotate-vlkuzun/src/somnotate_pipeline/run_complete_pipeline.py
"""
Somnotate Complete Pipeline Script

This script runs the complete Somnotate pipeline, including:
1. MAT to CSV conversion
2. EDF and Visbrain format generation
3. Path sheet generation
4. Signal preprocessing
5. Automated sleep stage annotation
6. Sleep state proportion analysis

Author: GitHub Copilot
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
import pyedflib
import matplotlib.pyplot as plt
import pickle
import logging
from datetime import datetime
from functools import partial
from pathlib import Path

# Try to import required modules from somnotate_pipeline
try:
    # Adjust path to include somnotate_pipeline directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    sys.path.append(os.path.join(parent_dir, 'src/somnotate_pipeline'))
    
    # Import required functions from pipeline modules
    from mat_to_csv import mat_to_csv
    from make_path_sheet import make_train_and_test_sheet
    from preprocess_signals import preprocess
    from data_io import (
        load_dataframe, check_dataframe, load_raw_signals, 
        load_preprocessed_signals, export_hypnogram, 
        export_preprocessed_signals, export_review_intervals
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure the somnotate_pipeline directory is in your PYTHONPATH")
    sys.exit(1)

# Try to import from somnotate package
try:
    from lspopt import spectrogram_lspopt
    from somnotate._utils import (
        robust_normalize, convert_state_vector_to_state_intervals, 
        _get_intervals
    )
    from somnotate._automated_state_annotation import StateAnnotator
    
    # Create partial function for spectrogram generation
    get_spectrogram = partial(spectrogram_lspopt, c_parameter=20.)
except ImportError as e:
    print(f"Error importing from somnotate package: {e}")
    print("Please make sure somnotate is installed")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("somnotate_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

def setup_output_directories(base_directory, dataset_type):
    """
    Create all necessary output directories for the pipeline.
    
    Args:
        base_directory (str): Base directory for the dataset
        dataset_type (str): Type of dataset ('train', 'test', or 'to_score')
    
    Returns:
        dict: Dictionary of paths to all created directories
    """
    directories = {
        'csv': os.path.join(base_directory, f"{dataset_type}_set", f"{dataset_type}_csv_files"),
        'edf': os.path.join(base_directory, f"{dataset_type}_set", 'edfs'),
        'manual_annotation': os.path.join(base_directory, f"{dataset_type}_set", f"{dataset_type}_manual_annotation"),
        'preprocessed': os.path.join(base_directory, f"{dataset_type}_set", "preprocessed_signals"),
        'automated_annotation': os.path.join(base_directory, f"{dataset_type}_set", "automated_annotation"),
        'review_intervals': os.path.join(base_directory, f"{dataset_type}_set", "int"),
        'figures': os.path.join(base_directory, f"{dataset_type}_set", "output_figures")
    }
    
    # Create each directory if it doesn't exist
    for dir_name, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return directories

def generate_edf_and_visbrain_formats(mouse_ids, sessions, recordings, extra_info, 
                                      dataset_type, base_directory, sample_frequency):
    """
    Generate EDF and Visbrain stage duration format files from CSV files.
    
    Args:
        mouse_ids (list): List of mouse IDs
        sessions (list): List of session IDs
        recordings (list): List of recording IDs
        extra_info (str): Additional information for file naming
        dataset_type (str): Type of dataset ('train', 'test', or 'to_score')
        base_directory (str): Base directory path
        sample_frequency (float): Sampling frequency in Hz
    """
    # Define output directories
    csv_input_dir = os.path.join(base_directory, f"{dataset_type}_set", f"{dataset_type}_csv_files")
    edf_output_dir = os.path.join(base_directory, f"{dataset_type}_set", 'edfs')
    annotations_output_dir = os.path.join(base_directory, f"{dataset_type}_set", f"{dataset_type}_manual_annotation")
    
    # Ensure directories exist
    os.makedirs(edf_output_dir, exist_ok=True)
    os.makedirs(annotations_output_dir, exist_ok=True)

    # Process each CSV file and generate output
    for mouse_id in mouse_ids:
        for session in sessions:
            for recording in recordings:
                # Prepare the base filename for the CSV file
                base_filename = f"{mouse_id}_{session}_{recording}"
                if extra_info:
                    csv_file = os.path.join(csv_input_dir, f"{base_filename}_{extra_info}.csv")
                else:
                    csv_file = os.path.join(csv_input_dir, f"{base_filename}.csv")
                
                # Prepare the base filename for EDF and Visbrain files
                if extra_info:
                    edf_file = os.path.join(edf_output_dir, f"output_{base_filename}_{extra_info}.edf")
                    visbrain_file = os.path.join(annotations_output_dir, f"annotations_visbrain_{base_filename}_{extra_info}.txt")
                else:
                    edf_file = os.path.join(edf_output_dir, f"output_{base_filename}.edf")
                    visbrain_file = os.path.join(annotations_output_dir, f"annotations_visbrain_{base_filename}.txt")

                if not os.path.isfile(csv_file):
                    logger.warning(f"File not found: {csv_file}")
                    continue
                if os.path.exists(edf_file) and not args.overwrite:
                    logger.info(f"EDF file already exists: {edf_file}")
                    continue
            
                logger.info(f"Processing file: {csv_file}")
                df = pd.read_csv(csv_file)

                # Extract EEG and EMG data
                eeg1_data = df["EEG1"].to_numpy()
                eeg2_data = df["EEG2"].to_numpy()
                emg_data = df["EMG"].to_numpy()

                # Combine all data
                all_data = np.array([eeg1_data, eeg2_data, emg_data])

                # Create an EDF file
                f = pyedflib.EdfWriter(edf_file, len(all_data), file_type=pyedflib.FILETYPE_EDFPLUS)

                # Define EDF header information
                labels = ["EEG1", "EEG2", "EMG"]
                for i, label in enumerate(labels):
                    signal_info = {
                        'label': label,
                        'dimension': 'uV',
                        'sample_frequency': sample_frequency,
                        'physical_min': np.min(all_data[i]),
                        'physical_max': np.max(all_data[i]),
                        'digital_min': -32768,
                        'digital_max': 32767,
                        'transducer': '',
                        'prefilter': ''
                    }
                    f.setSignalHeader(i, signal_info)

                # Write EEG and EMG data to the EDF file
                f.writeSamples(all_data)
                f.close()

                # Prepare annotations in Visbrain stage duration format
                annotations = [(0, 10, "Undefined")]
                current_stage = None
                start_time = 10 / sample_frequency

                for i, label in enumerate(df["sleepStage"]):
                    current_time = i / sample_frequency
                    if label != current_stage:
                        if current_stage is not None:
                            annotations.append((start_time, current_time, current_stage))
                        current_stage = label
                        start_time = current_time
                annotations.append((start_time, len(df) / sample_frequency, current_stage))

                # Write annotations to a text file
                last_time_value = annotations[-1][1]
                with open(visbrain_file, "w") as f:
                    f.write(f"*Duration_sec    {last_time_value}\n")
                    f.write("*Datafile\tUnspecified\n")
                    for start, end, stage in annotations:
                        stage_label = {1: "awake", 2: "non-REM", 3: "REM", 
                                       4: 'ambiguous', 5: 'doubt'}.get(stage, "Undefined")
                        f.write(f"{stage_label}    {end}\n")

                logger.info(f"EDF file and annotations created successfully for {mouse_id}, {session}, {recording}")

def export_intervals_with_low_confidence(file_path, state_probability, threshold=0.99, time_resolution=1):
    """
    Export intervals with low confidence for manual review.
    
    Args:
        file_path (str): Path to save intervals
        state_probability (numpy.ndarray): Array of state probabilities
        threshold (float): Confidence threshold (default: 0.99)
        time_resolution (int): Time resolution in seconds (default: 1)
    """
    intervals = _get_intervals(state_probability < threshold)
    scores = [len(state_probability[start:stop]) - np.sum(state_probability[start:stop]) 
              for start, stop in intervals]
    intervals = [(start * time_resolution, stop * time_resolution) for start, stop in intervals]
    notes = ['probability below threshold' for _ in intervals]
    export_review_intervals(file_path, intervals, scores, notes)

def analyze_sleep_state_proportions(annotation_file, start_time_sec, end_time_sec, output_file=None):
    """
    Analyze the proportion of different sleep states within a specified time window.
    
    Args:
        annotation_file (str): Path to the annotation file in Visbrain format
        start_time_sec (float): Start time of the analysis window in seconds
        end_time_sec (float): End time of the analysis window in seconds
        output_file (str, optional): Path to save the results plot
        
    Returns:
        dict: Dictionary containing the proportion of each sleep state and total duration
    """
    # Read the annotation file
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    
    # Extract the duration from the header
    duration_line = lines[0].strip()
    if duration_line.startswith('*Duration_sec'):
        total_duration = float(duration_line.split()[1])
    else:
        raise ValueError("Invalid annotation file format: missing duration header")
    
    # Skip header lines
    data_lines = [line for line in lines if not line.startswith('*')]
    
    # Parse annotations
    end_times = []
    states = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            states.append(parts[0])
            end_times.append(float(parts[1]))
    
    # Calculate state intervals
    intervals = []
    start_times = [0] + end_times[:-1]
    for state, start, end in zip(states, start_times, end_times):
        intervals.append((state, start, end))
    
    # Filter intervals to the specified time window
    filtered_intervals = []
    for state, start, end in intervals:
        # Skip intervals outside the time window
        if end <= start_time_sec or start >= end_time_sec:
            continue
        
        # Clip interval to the time window
        clipped_start = max(start, start_time_sec)
        clipped_end = min(end, end_time_sec)
        filtered_intervals.append((state, clipped_start, clipped_end))
    
    # Calculate total duration of each state
    state_durations = {}
    for state, start, end in filtered_intervals:
        duration = end - start
        state_durations[state] = state_durations.get(state, 0) + duration
    
    # Calculate total window duration and proportions
    window_duration = end_time_sec - start_time_sec
    state_proportions = {state: duration / window_duration 
                        for state, duration in state_durations.items()}
    
    # Add total durations to results
    state_proportions['window_duration'] = window_duration
    for state, duration in state_durations.items():
        state_proportions[f'{state}_duration'] = duration
    
    # Generate visualization if output file is specified
    if output_file:
        # Extract just the proportions for the pie chart
        labels = []
        sizes = []
        for state in state_proportions:
            if not state.endswith('_duration') and state != 'window_duration':
                labels.append(state)
                sizes.append(state_proportions[state])
        
        # Create custom colors for standard states
        colors = []
        for label in labels:
            if label == 'awake':
                colors.append('gold')
            elif label == 'non-REM':
                colors.append('royalblue')
            elif label == 'REM':
                colors.append('crimson')
            else:
                colors.append('gray')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85
        )
        
        # Draw a white circle at the center to create a donut chart
        centre_circle = plt.Circle((0, 0), 0.60, fc='white')
        fig.gca().add_artist(centre_circle)
        
        # Add title and annotation
        ax.set_title(f'Sleep State Proportions ({start_time_sec:.2f}s - {end_time_sec:.2f}s)', size=14)
        plt.text(0, 0, f"Total: {state_proportions['window_duration']:.2f}s", ha='center', va='center', fontsize=12)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_file)
        logger.info(f"Saved proportions visualization to {output_file}")
        plt.close(fig)
    
    return state_proportions

def batch_analyze_sleep_proportions(file_list, time_windows, output_folder, generate_plots=True):
    """
    Analyze sleep state proportions for multiple files and time windows.
    
    Args:
        file_list (list): List of paths to annotation files
        time_windows (list): List of (start_time, end_time) tuples in seconds
        output_folder (str): Path to save results and plots
        generate_plots (bool): Whether to generate visualization plots
        
    Returns:
        pandas.DataFrame: DataFrame containing the results
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare output CSV path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_folder, f"sleep_proportions_{timestamp}.csv")
    
    results = []
    
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        
        for start_time, end_time in time_windows:
            try:
                # Skip invalid time windows
                if start_time >= end_time:
                    logger.warning(f"Skipping invalid time window: {start_time}-{end_time}")
                    continue
                
                logger.info(f"Processing {file_name}: {start_time}s to {end_time}s")
                
                # Generate output plot path if needed
                output_plot = None
                if generate_plots:
                    plot_filename = f"sleep_proportions_{file_name.replace('.txt', '')}_{int(start_time)}_{int(end_time)}.png"
                    output_plot = os.path.join(output_folder, plot_filename)
                
                # Calculate proportions
                proportions = analyze_sleep_state_proportions(
                    file_path, start_time, end_time, output_plot)
                
                # Create a row for this result
                result_row = {
                    'file_name': file_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'window_duration': proportions['window_duration']
                }
                
                # Add proportions for each state
                for state, proportion in proportions.items():
                    if state not in ['window_duration'] and not state.endswith('_duration'):
                        result_row[f'{state}_proportion'] = proportion
                        result_row[f'{state}_duration'] = proportions.get(f'{state}_duration', 0)
                
                results.append(result_row)
                
            except Exception as e:
                logger.error(f"Error processing {file_name}, window {start_time}-{end_time}: {str(e)}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    if len(results) > 0:
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")
    else:
        logger.warning("No results to save")
    
    return results_df

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run complete Somnotate pipeline')
    
    # General parameters
    parser.add_argument('--dataset-type', type=str, choices=['train', 'test', 'to_score'], 
                        default='to_score', help='Dataset type')
    parser.add_argument('--base-dir', type=str, required=True, 
                        help='Base directory for somnotate data')
    parser.add_argument('--sampling-rate', type=float, default=512.0, 
                        help='Sampling rate in Hz')
    parser.add_argument('--sleep-stage-res', type=int, default=10, 
                        help='Sleep stage resolution in seconds')
    parser.add_argument('--mouse-ids', type=str, required=True, 
                        help='Comma-separated mouse IDs')
    parser.add_argument('--sessions', type=str, required=True, 
                        help='Comma-separated session IDs')
    parser.add_argument('--recordings', type=str, required=True, 
                        help='Comma-separated recording IDs')
    parser.add_argument('--extra-info', type=str, default='', 
                        help='Extra information for file naming')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Overwrite existing files')
    parser.add_argument('--steps', type=str, default='1,2,3,4,5,6', 
                        help='Comma-separated list of steps to run (1-6)')
    
    # Step 1: MAT to CSV
    parser.add_argument('--mat-files', type=str, default='', 
                        help='Comma-separated paths to MAT files')
    
    # Step 5: Automated Annotation
    parser.add_argument('--model-path', type=str, default='', 
                        help='Path to trained model (.pickle file)')
    
    # Step 6: Sleep State Proportions
    parser.add_argument('--analysis-windows', type=str, default='', 
                        help='Comma-separated list of time windows to analyze (start1-end1,start2-end2,...)')
    
    # Generate plots
    parser.add_argument('--generate-plots', action='store_true', 
                        help='Generate visualization plots')
    
    return parser.parse_args()

def run_mat_to_csv(args):
    """Run MAT to CSV conversion step."""
    logger.info("STEP 1: MAT to CSV Conversion")
    
    # Extract file paths
    file_paths = []
    if args.mat_files:
        for path in args.mat_files.split(','):
            path = path.strip()
            if os.path.isfile(path) and path.endswith('.mat'):
                file_paths.append(path)
            else:
                logger.warning(f"Invalid MAT file: {path}")
    
    if not file_paths:
        logger.error("No valid MAT files provided")
        return False
    
    # Setup output directory
    output_dir = os.path.join(args.base_dir, f"{args.dataset_type}_set", 
                              f"{args.dataset_type}_csv_files")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info(f"Converting {len(file_paths)} MAT files to CSV")
        mat_to_csv(file_paths, output_dir, args.sampling_rate, args.sleep_stage_res)
        logger.info("MAT to CSV conversion completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during MAT to CSV conversion: {str(e)}")
        return False

def run_edf_generation(args):
    """Run EDF and Visbrain format generation step."""
    logger.info("STEP 2: Generate EDF and Visbrain Format Files")
    
    mouse_ids = [m.strip() for m in args.mouse_ids.split(',')]
    sessions = [s.strip() for s in args.sessions.split(',')]
    recordings = [r.strip() for r in args.recordings.split(',')]
    
    try:
        generate_edf_and_visbrain_formats(
            mouse_ids,
            sessions,
            recordings,
            args.extra_info,
            args.dataset_type,
            args.base_dir,
            args.sampling_rate
        )
        logger.info("EDF and Visbrain format generation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during EDF and Visbrain format generation: {str(e)}")
        return False

def run_path_sheet_generation(args):
    """Run path sheet generation step."""
    logger.info("STEP 3: Generate Path Sheet")
    
    expected_output_path = os.path.join(args.base_dir, f"{args.dataset_type}_set", 
                                       f"{args.dataset_type}_sheet.csv")
    
    try:
        output_file = make_train_and_test_sheet(args.dataset_type, args.base_dir, args.sampling_rate)
        logger.info(f"Path sheet generation completed successfully: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error during path sheet generation: {str(e)}")
        return False

def run_signal_preprocessing(args):
    """Run signal preprocessing step."""
    logger.info("STEP 4: Preprocess Signals")
    
    path_sheet_path = os.path.join(args.base_dir, f"{args.dataset_type}_set", 
                                  f"{args.dataset_type}_sheet.csv")
    
    if not os.path.exists(path_sheet_path):
        logger.error(f"Path sheet not found at {path_sheet_path}")
        return False
    
    try:
        # Load the path sheet
        datasets = load_dataframe(path_sheet_path)
        logger.info(f"Loaded {len(datasets)} dataset(s) from {path_sheet_path}")
        
        # Check required columns
        required_columns = ['file_path_raw_signals', 'sampling_frequency_in_hz', 
                           'file_path_preprocessed_signals', 'eeg1_signal_label', 
                           'eeg2_signal_label', 'emg_signal_label']
        check_dataframe(datasets, columns=required_columns)
        
        # Define commonly used signal labels
        state_annotation_signals = ['eeg1_signal_label', 'eeg2_signal_label', 'emg_signal_label']
        
        # Set preprocessing parameters
        time_resolution = 1  # Default time resolution in seconds
        
        # Create output directory for plots if showing them
        if args.generate_plots:
            output_folder = os.path.join(os.path.dirname(path_sheet_path), 'output_figures')
            os.makedirs(output_folder, exist_ok=True)
        
        # Process each dataset in the path sheet
        for ii, (idx, dataset) in enumerate(datasets.iterrows()):
            logger.info(f"Processing {ii+1}/{len(datasets)}: {os.path.basename(dataset['file_path_raw_signals'])}")
            
            # Determine EDF signals to load
            signal_labels = [dataset[column_name] for column_name in state_annotation_signals]
            
            # Load data
            raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)
            
            # Create directory for preprocessed signals if it doesn't exist
            os.makedirs(os.path.dirname(dataset['file_path_preprocessed_signals']), exist_ok=True)
            
            # Preprocess each signal
            preprocessed_signals = []
            for signal in raw_signals.T:
                # Convert time_resolution_in_sec * sampling_frequency to integer for nperseg
                nperseg = int(time_resolution * dataset['sampling_frequency_in_hz'])
                
                # Get spectrogram
                frequencies, time, spectrogram = get_spectrogram(
                    signal,
                    fs=dataset['sampling_frequency_in_hz'],
                    nperseg=nperseg,
                    noverlap=0
                )
                
                # Filter frequency bands
                low_cut = 1.0
                high_cut = 90.0
                notch_low_cut = 45.0
                notch_high_cut = 55.0
                
                # exclude ill-determined frequencies
                mask = (frequencies >= low_cut) & (frequencies < high_cut)
                frequencies = frequencies[mask]
                spectrogram = spectrogram[mask]

                # exclude noise-contaminated frequencies around 50 Hz
                mask = (frequencies >= notch_low_cut) & (frequencies <= notch_high_cut)
                frequencies = frequencies[~mask]
                spectrogram = spectrogram[~mask]

                # Log transform and normalize
                spectrogram = np.log(spectrogram + 1)
                spectrogram = robust_normalize(spectrogram, p=5., axis=1, method='standard score')
                
                preprocessed_signals.append(spectrogram)
            
            # Generate plots if requested
            if args.generate_plots:
                fig, axes = plt.subplots(1+len(preprocessed_signals), 1, figsize=(12, 8), sharex=True)
                
                # Plot raw signals
                time_raw = np.arange(len(raw_signals)) / dataset['sampling_frequency_in_hz']
                if isinstance(axes, np.ndarray):
                    ax = axes[0]
                else:
                    ax = axes
                for i, signal in enumerate(raw_signals.T):
                    ax.plot(time_raw, signal, label=f"Channel {i+1}")
                ax.set_ylabel('Amplitude')
                ax.legend()
                
                # Plot spectrograms
                for i, signal in enumerate(preprocessed_signals):
                    if isinstance(axes, np.ndarray):
                        ax = axes[i+1]
                    else:
                        fig_new, ax = plt.subplots(figsize=(12, 4))
                    im = ax.imshow(
                        signal, 
                        aspect='auto', 
                        origin='lower', 
                        extent=[time[0], time[-1], frequencies[0], frequencies[-1]],
                        cmap='viridis'
                    )
                    ax.set_ylabel('Frequency (Hz)')
                    plt.colorbar(im, ax=ax, label='Power (normalized)')
                
                # Add common labels
                if isinstance(axes, np.ndarray):
                    axes[-1].set_xlabel('Time (s)')
                else:
                    axes.set_xlabel('Time (s)')
                
                # Save figure
                file_basename = os.path.basename(dataset['file_path_raw_signals'])
                fig_path = os.path.join(output_folder, f'preprocessed_signals_{file_basename}.png')
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close(fig)
                logger.info(f"  Saved figure to {fig_path}")
            
            # Concatenate spectrograms and save
            preprocessed_signals = np.concatenate([signal.T for signal in preprocessed_signals], axis=1)
            export_preprocessed_signals(dataset['file_path_preprocessed_signals'], preprocessed_signals)
            logger.info(f"  Saved preprocessed signals to {dataset['file_path_preprocessed_signals']}")
        
        logger.info("Signal preprocessing completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during signal preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_sleep_stage_annotation(args):
    """Run automated sleep stage annotation step."""
    logger.info("STEP 5: Automated Sleep Stage Annotation")
    
    if not args.model_path:
        logger.error("Model path is required for sleep stage annotation")
        return False
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found at {args.model_path}")
        return False
    
    path_sheet_path = os.path.join(args.base_dir, f"{args.dataset_type}_set", 
                                  f"{args.dataset_type}_sheet.csv")
    
    if not os.path.exists(path_sheet_path):
        logger.error(f"Path sheet not found at {path_sheet_path}")
        return False
    
    try:
        # Load the path sheet
        datasets = load_dataframe(path_sheet_path)
        logger.info(f"Loaded {len(datasets)} dataset(s) from {path_sheet_path}")
        
        # Check required columns
        required_columns = [
            'file_path_preprocessed_signals', 
            'file_path_automated_state_annotation', 
            'file_path_review_intervals'
        ]
        if args.generate_plots:
            required_columns.extend(['file_path_raw_signals', 'sampling_frequency_in_hz'])
            required_columns.extend(['eeg1_signal_label', 'eeg2_signal_label', 'emg_signal_label'])
        
        check_dataframe(datasets, columns=required_columns)
        
        # Define state mappings
        int_to_state = {
            1: "awake",
            2: "non-REM",
            3: "REM",
            4: "ambiguous",
            5: "doubt"
        }
        
        # Time resolution in seconds
        time_resolution = 1
        
        # Load the trained model
        logger.info(f"Loading trained model from {args.model_path}")
        annotator = StateAnnotator()
        annotator.load(args.model_path)
        
        # Process each dataset
        for ii, (idx, dataset) in enumerate(datasets.iterrows()):
            logger.info(f"Processing {ii+1}/{len(datasets)}: {os.path.basename(dataset['file_path_preprocessed_signals'])}")
            
            # Load preprocessed signals
            try:
                signal_array = load_preprocessed_signals(dataset['file_path_preprocessed_signals'])
            except Exception as e:
                logger.error(f"  Error loading preprocessed signals: {e}")
                continue
            
            # Create output directories if needed
            for path_key in ['file_path_automated_state_annotation', 'file_path_review_intervals']:
                directory = os.path.dirname(dataset[path_key])
                os.makedirs(directory, exist_ok=True)
            
            # Predict states and probabilities
            logger.info("  Annotating sleep states...")
            predicted_state_vector = annotator.predict(signal_array)
            state_probability = annotator.predict_proba(signal_array)
            
            # Convert to intervals and export
            predicted_states, predicted_intervals = convert_state_vector_to_state_intervals(
                predicted_state_vector, 
                mapping=int_to_state, 
                time_resolution=time_resolution
            )
            
            # Export hypnogram and review intervals
            export_hypnogram(dataset['file_path_automated_state_annotation'], 
                             predicted_states, 
                             predicted_intervals)
            logger.info(f"  Saved hypnogram to {dataset['file_path_automated_state_annotation']}")
            
            export_intervals_with_low_confidence(
                dataset['file_path_review_intervals'],
                state_probability,
                threshold=0.99,
                time_resolution=time_resolution
            )
            logger.info(f"  Saved review intervals to {dataset['file_path_review_intervals']}")
            
            # Generate visualization if requested
            if args.generate_plots:
                logger.info("  Generating visualization...")
                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                
                # Plot raw signals
                signal_labels = [dataset[col] for col in ['eeg1_signal_label', 'eeg2_signal_label', 'emg_signal_label']]
                raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)
                
                # Time array for raw signals
                time_raw = np.arange(len(raw_signals)) / dataset['sampling_frequency_in_hz']
                
                # Plot raw signals
                for i, (signal, label) in enumerate(zip(raw_signals.T, signal_labels)):
                    axes[0].plot(time_raw, signal, label=label)
                axes[0].set_title('Raw Signals')
                axes[0].set_ylabel('Amplitude')
                axes[0].legend()
                
                # Plot prediction probabilities
                time_prob = np.arange(len(state_probability)) * time_resolution
                axes[1].plot(time_prob, state_probability)
                axes[1].set_title('Prediction Confidence')
                axes[1].set_ylabel('Confidence')
                axes[1].axhline(y=0.99, color='r', linestyle='--', label='Threshold')
                axes[1].legend()
                
                # Plot predicted states
                yticklabels = list(int_to_state.values())
                yticks = list(range(1, len(yticklabels) + 1))
                
                # Convert states to numeric for plotting
                state_nums = []
                for state in predicted_states:
                    for num, label in int_to_state.items():
                        if state == label:
                            state_nums.append(num)
                            break
                
                for i, (state, (start, end)) in enumerate(zip(state_nums, predicted_intervals)):
                    axes[2].fill_between([start, end], [state, state], [state-0.9, state-0.9], alpha=0.7)
                
                axes[2].set_yticks(yticks)
                axes[2].set_yticklabels(yticklabels)
                axes[2].set_title('Predicted Sleep Stages')
                axes[2].set_ylabel('State')
                axes[2].set_xlabel('Time (s)')
                
                plt.tight_layout()
                
                # Save figure
                output_folder = os.path.join(os.path.dirname(path_sheet_path), 'output_figures')
                os.makedirs(output_folder, exist_ok=True)
                
                file_basename = os.path.basename(dataset['file_path_preprocessed_signals']).replace('.h5', '')
                fig_path = os.path.join(output_folder, f'annotation_{file_basename}.png')
                plt.savefig(fig_path)
                plt.close(fig)
                logger.info(f"  Saved figure to {fig_path}")
        
        logger.info("Sleep stage annotation completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error during sleep stage annotation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_sleep_state_proportion_analysis(args):
    """Run sleep state proportion analysis step."""
    logger.info("STEP 6: Calculate Sleep State Proportions")
    
    # Find annotation files to analyze
    annotation_dir = os.path.join(args.base_dir, f"{args.dataset_type}_set", "automated_annotation")
    
    if not os.path.exists(annotation_dir):
        logger.error(f"Annotation directory not found at {annotation_dir}")
        return False
    
    # Get all annotation files
    annotation_files = []
    for file in os.listdir(annotation_dir):
        if file.endswith('.txt') and not file.startswith('.'):
            annotation_files.append(os.path.join(annotation_dir, file))
    
    if not annotation_files:
        logger.error("No annotation files found to analyze")
        return False
    
    # Parse time windows
    time_windows = []
    if args.analysis_windows:
        for window in args.analysis_windows.split(','):
            try:
                start, end = map(float, window.split('-'))
                if start < end:
                    time_windows.append((start, end))
                else:
                    logger.warning(f"Invalid time window {window}: start time must be less than end time")
            except ValueError:
                logger.warning(f"Invalid time window format: {window}. Expected format: start-end")
    
    # If no windows specified, use default windows
    if not time_windows:
        # Use 1-hour windows for first 6 hours
        for i in range(6):
            time_windows.append((i * 3600, (i + 1) * 3600))
    
    # Output directory for results
    output_folder = os.path.join(args.base_dir, f"{args.dataset_type}_set", "sleep_proportions")
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        logger.info(f"Analyzing {len(annotation_files)} files with {len(time_windows)} time windows")
        results_df = batch_analyze_sleep_proportions(
            annotation_files, 
            time_windows, 
            output_folder,
            generate_plots=args.generate_plots
        )
        
        # Display summary of results
        if not results_df.empty:
            logger.info(f"Analysis completed with {len(results_df)} results")
            
            # Save summary statistics
            summary_file = os.path.join(output_folder, "summary_statistics.txt")
            with open(summary_file, 'w') as f:
                f.write("Sleep State Proportion Analysis Summary\n")
                f.write("======================================\n\n")
                
                # Write aggregate statistics
                f.write("Aggregate Statistics Across All Windows:\n")
                f.write("---------------------------------\n")
                
                # Get columns with proportions
                prop_cols = [col for col in results_df.columns if col.endswith('_proportion')]
                dur_cols = [col for col in results_df.columns if col.endswith('_duration')]
                
                # Calculate and write mean proportions
                f.write("Mean Proportions:\n")
                for col in prop_cols:
                    state = col.replace('_proportion', '')
                    mean_val = results_df[col].mean() * 100  # Convert to percentage
                    f.write(f"{state}: {mean_val:.2f}%\n")
                
                # Calculate and write total durations
                f.write("\nTotal Durations:\n")
                for col in dur_cols:
                    state = col.replace('_duration', '')
                    total_dur = results_df[col].sum()
                    f.write(f"{state}: {total_dur:.2f} seconds ({total_dur/3600:.2f} hours)\n")
                
                # Write window-specific statistics
                f.write("\n\nStatistics by Time Window:\n")
                f.write("------------------------\n")
                
                for window in time_windows:
                    start, end = window
                    window_df = results_df[(results_df['start_time'] == start) & (results_df['end_time'] == end)]
                    
                    if not window_df.empty:
                        f.write(f"\nWindow {start}-{end} seconds ({start/3600:.2f}-{end/3600:.2f} hours):\n")
                        
                        for col in prop_cols:
                            state = col.replace('_proportion', '')
                            mean_val = window_df[col].mean() * 100  # Convert to percentage
                            f.write(f"{state}: {mean_val:.2f}%\n")
            
            logger.info(f"Saved summary statistics to {summary_file}")
            return True
        else:
            logger.warning("No results generated from analysis")
            return False
        
    except Exception as e:
        logger.error(f"Error during sleep state proportion analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert steps to a list of integers
    try:
        steps_to_run = [int(step) for step in args.steps.split(',')]
    except ValueError:
        logger.error("Invalid steps format. Please use comma-separated integers (1-6)")
        return 1
    
    # Validate steps
    valid_steps = list(range(1, 7))
    if any(step not in valid_steps for step in steps_to_run):
        logger.error(f"Invalid step number. Valid steps are: {valid_steps}")
        return 1
    
    # Setup output directories
    try:
        directories = setup_output_directories(args.base_dir, args.dataset_type)
    except Exception as e:
        logger.error(f"Error setting up output directories: {str(e)}")
        return 1
    
    # Run each requested step
    results = []
    
    # Step 1: MAT to CSV conversion
    if 1 in steps_to_run:
        success = run_mat_to_csv(args)
        results.append(("MAT to CSV Conversion", success))
    
    # Step 2: EDF and Visbrain format generation
    if 2 in steps_to_run:
        success = run_edf_generation(args)
        results.append(("EDF and Visbrain Generation", success))
    
    # Step 3: Path sheet generation
    if 3 in steps_to_run:
        success = run_path_sheet_generation(args)
        results.append(("Path Sheet Generation", success))
    
    # Step 4: Signal preprocessing
    if 4 in steps_to_run:
        success = run_signal_preprocessing(args)
        results.append(("Signal Preprocessing", success))
    
    # Step 5: Automated sleep stage annotation
    if 5 in steps_to_run:
        success = run_sleep_stage_annotation(args)
        results.append(("Sleep Stage Annotation", success))
    
    # Step 6: Sleep state proportion analysis
    if 6 in steps_to_run:
        success = run_sleep_state_proportion_analysis(args)
        results.append(("Sleep State Proportion Analysis", success))
    
    # Display summary of results
    logger.info("\n" + "=" * 50)
    logger.info("Pipeline Execution Summary")
    logger.info("=" * 50)
    
    all_success = True
    for step_name, success in results:
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{step_name}: {status}")
        if not success:
            all_success = False
    
    if all_success:
        logger.info("\nAll steps completed successfully!")
        return 0
    else:
        logger.warning("\nSome steps failed. Please check the log for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())