#!/usr/bin/env python

"""
Somnotate Full Pipeline

This script combines all steps of the somnotate pipeline:
1. Convert MAT files to CSV format
2. Generate EDF and Visbrain format files for visualization and analysis
3. Generate path sheet for organizing files
4. Preprocess signals for sleep stage analysis
5. Automated sleep stage annotation
6. Sleep state proportion analysis

Run the script with --help for usage information.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pyedflib
from functools import partial

# Add somnotate_pipeline to path if not there already
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import functions from pipeline modules
from mat_to_csv import mat_to_csv
from make_path_sheet import make_train_and_test_sheet
from preprocess_signals import preprocess, get_spectrogram

# Import utility functions from somnotate
try:
    from somnotate._utils import robust_normalize, convert_state_vector_to_state_intervals, _get_intervals
    from somnotate._automated_state_annotation import StateAnnotator
    from data_io import (
        load_dataframe, check_dataframe, load_raw_signals, 
        export_preprocessed_signals, load_preprocessed_signals,
        export_hypnogram, export_review_intervals
    )
except ImportError:
    print("Error: somnotate package not found. Make sure it's properly installed.")
    sys.exit(1)


def print_step_header(step_num, step_name):
    """Print a formatted header for each pipeline step."""
    print(f"\nSTEP {step_num}: {step_name}")
    print("-" * (len(step_name) + 8))


def collect_common_parameters():
    """Collect common parameters that will be used throughout the pipeline."""
    print("Enter the common parameters that will be used throughout the pipeline:")
    print("-" * 70)
    
    # Dataset type
    dataset_type = input("Enter dataset type ('train', 'test', or 'to_score'): ")
    
    # Base directory
    base_directory = input(f"Enter the base somnotate directory path, without quotes (e.g., Z:/somnotate): ")
    
    # Sampling rate
    sampling_rate = float(input("Enter the sampling rate in Hz (e.g., 512.0): "))
    
    # Sleep stage resolution
    sleep_stage_resolution = int(input("Enter the sleep stage resolution in seconds (e.g., 10): "))
    
    # Mouse, session, recording information
    print("\nEnter subject and recording information (format: comma-separated values without spaces)")
    mouse_ids = input("Enter mouse IDs (e.g., sub-001,sub-002): ").split(',')
    sessions = input("Enter session IDs (e.g., ses-01,ses-02): ").split(',')
    recordings = input("Enter recording IDs (e.g., recording-01,recording-02): ").split(',')
    extra_info = input("Enter any extra details about the recording (leave blank if not applicable): ").strip()
    
    # Define derived paths
    output_directory_path = os.path.join(base_directory, f"{dataset_type}_set", f"{dataset_type}_csv_files")
    path_sheet_path = os.path.join(base_directory, f"{dataset_type}_set", f"{dataset_type}_sheet.csv")
    
    # Collect MAT file paths for Step 1 (MAT to CSV conversion)
    print("\nFor MAT to CSV conversion (Step 1), enter the full paths of .mat files:")
    print("(Press Enter on an empty line to finish)")
    
    mat_file_paths = []
    while True:
        file_path = input("Enter MAT file path: ")
        if file_path == "":
            break
        if os.path.isfile(file_path) and file_path.endswith('.mat'):
            mat_file_paths.append(file_path)
        else:
            print("Invalid file path. Please enter a valid .mat file path.")
    
    # Also collect the model path for Step 5 (Automated Sleep Stage Annotation)
    model_path = input("\nFor Automated Sleep Stage Annotation (Step 5), enter the path to the trained model (.pickle file): ")
    
    # Note: We don't collect annotation file path for Step 6 here
    # as it will likely be created during Step 5
    
    # Print summary
    print("\nInput Summary:")
    print(f"Dataset type: {dataset_type}")
    print(f"Base directory: {base_directory}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Sleep stage resolution: {sleep_stage_resolution} seconds")
    print(f"Mouse IDs: {', '.join(mouse_ids)}")
    print(f"Session IDs: {', '.join(sessions)}")
    print(f"Recording IDs: {', '.join(recordings)}")
    print(f"Extra info: {extra_info if extra_info else 'None'}")
    print(f"CSV output directory: {output_directory_path}")
    print(f"Path sheet location: {path_sheet_path}")
    print(f"MAT files to convert: {len(mat_file_paths)}")
    
    return {
        'dataset_type': dataset_type,
        'base_directory': base_directory,
        'sampling_rate': sampling_rate,
        'sleep_stage_resolution': sleep_stage_resolution,
        'mouse_ids': mouse_ids,
        'sessions': sessions,
        'recordings': recordings,
        'extra_info': extra_info,
        'output_directory_path': output_directory_path,
        'path_sheet_path': path_sheet_path,
        'mat_file_paths': mat_file_paths,
        'model_path': model_path
        # Removed annotation_file, start_time, and end_time as they'll be collected later
    }


def execute_step_one(params):
    """Execute Step 1: Convert MAT files to CSV."""
    print_step_header(1, "MAT to CSV Conversion")
    
    print(f"Using dataset type: {params['dataset_type']}")
    print(f"Using output directory: {params['output_directory_path']}")
    print(f"Using sampling rate: {params['sampling_rate']} Hz")
    print(f"Using sleep stage resolution: {params['sleep_stage_resolution']} seconds")
    
    # Use file paths already collected in common parameters
    file_paths = params['mat_file_paths']
    
    # If no files were provided during parameter collection, ask now
    if not file_paths:
        print("\nNo MAT files were specified during initial parameter collection.")
        choice = input("Would you like to specify MAT files now? (y/n): ").strip().lower()
        
        if choice == 'y':
            print("\nEnter the full paths of .mat files to convert (press Enter on an empty line to finish):")
            while True:
                file_path = input("Enter file path: ")
                if file_path == "":
                    break
                if os.path.isfile(file_path) and file_path.endswith('.mat'):
                    file_paths.append(file_path)
                else:
                    print("Invalid file path. Please enter a valid .mat file path.")
    
    # Print summary of files to process
    print(f"\nFiles to process: {len(file_paths)}")
    for i, path in enumerate(file_paths):
        print(f"  {i+1}. {path}")
    
    # Confirm before proceeding
    proceed = input("\nProceed with MAT to CSV conversion? (y/n): ").strip().lower()
    
    if proceed == 'y' and file_paths and params['output_directory_path']:
        try:
            print("Starting MAT to CSV conversion process...")
            
            # Create the output directory if it doesn't exist
            if not os.path.exists(params['output_directory_path']):
                os.makedirs(params['output_directory_path'])
                print(f"Created output directory: {params['output_directory_path']}")
            
            # Execute the conversion
            mat_to_csv(file_paths, params['output_directory_path'], params['sampling_rate'], params['sleep_stage_resolution'])
            
            print("\nMAT to CSV conversion completed successfully!")
            print(f"CSV files are saved to: {params['output_directory_path']}")
            
            # List the created output files
            output_files = [f for f in os.listdir(params['output_directory_path']) if f.endswith('.csv')]
            print(f"\nCreated {len(output_files)} CSV files:")
            for i, file in enumerate(output_files):
                print(f"  {i+1}. {file}")
            
            return True
        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("MAT to CSV conversion cancelled or missing required inputs.")
        return False


def generate_edf_and_visbrain_formats(mouse_ids, sessions, recordings, extra_info, test_train_or_to_score, base_directory, sample_frequency):
    """
    Generate EDF and Visbrain stage duration format files from CSV files, for EEG/EMG data and sleep stage annotations.
    """
    # Define output directories
    csv_input_dir = os.path.join(base_directory, f"{test_train_or_to_score}_set/{test_train_or_to_score}_csv_files")
    edf_output_dir = os.path.join(base_directory, f"{test_train_or_to_score}_set", 'edfs')
    annotations_output_dir = os.path.join(base_directory, f"{test_train_or_to_score}_set", f"{test_train_or_to_score}_manual_annotation")
    
    if not os.path.exists(edf_output_dir):
        os.makedirs(edf_output_dir)
    if not os.path.exists(annotations_output_dir):
        os.makedirs(annotations_output_dir)

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
                    print(f"File not found: {csv_file}")
                    continue
                if os.path.exists(edf_file):
                    print(f"EDF file already exists: {edf_file}")
                    continue
            
                print(f"Processing file: {csv_file}")
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
                        stage_label = {1: "awake", 2: "non-REM", 3: "REM", 4: 'ambiguous', 5: 'doubt'}.get(stage, "Undefined")
                        f.write(f"{stage_label}    {end}\n")

                print(f"EDF file and annotations created successfully for {mouse_id}, {session}, {recording} with extra info '{extra_info}'.")


def execute_step_two(params):
    """Execute Step 2: Generate EDF and Visbrain Format Files."""
    print_step_header(2, "Generate EDF and Visbrain Format Files")
    
    print(f"Using dataset type: {params['dataset_type']}")
    print(f"Using base directory: {params['base_directory']}")
    print(f"Using sampling rate: {params['sampling_rate']} Hz")
    print(f"Using mouse IDs: {', '.join(params['mouse_ids'])}")
    print(f"Using session IDs: {', '.join(params['sessions'])}")
    print(f"Using recording IDs: {', '.join(params['recordings'])}")
    print(f"Using extra info: {params['extra_info'] if params['extra_info'] else 'None'}")
    
    # Calculate expected output paths for verification
    edf_output_dir = os.path.join(params['base_directory'], f"{params['dataset_type']}_set", 'edfs')
    annotations_output_dir = os.path.join(params['base_directory'], f"{params['dataset_type']}_set", f"{params['dataset_type']}_manual_annotation")
    
    print(f"\nEDF files will be saved to: {edf_output_dir}")
    print(f"Annotation files will be saved to: {annotations_output_dir}")
    
    # Confirm before proceeding
    proceed = input("\nProceed with EDF and Visbrain format generation? (y/n): ").strip().lower()
    
    if proceed == 'y':
        try:
            print("Starting EDF and Visbrain format generation...")
            
            # Create directories if they don't exist
            if not os.path.exists(edf_output_dir):
                os.makedirs(edf_output_dir)
                print(f"Created directory: {edf_output_dir}")
                
            if not os.path.exists(annotations_output_dir):
                os.makedirs(annotations_output_dir)
                print(f"Created directory: {annotations_output_dir}")
                
            # Call the function
            generate_edf_and_visbrain_formats(
                params['mouse_ids'],
                params['sessions'],
                params['recordings'],
                params['extra_info'],
                params['dataset_type'],
                params['base_directory'],
                params['sampling_rate']
            )
            
            print("\nEDF and Visbrain format generation completed!")
            print(f"EDF files saved to: {edf_output_dir}")
            print(f"Visbrain annotations saved to: {annotations_output_dir}")
            
            return True
        except Exception as e:
            print(f"Error during EDF and Visbrain format generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("EDF and Visbrain format generation cancelled.")
        return False


def execute_step_three(params):
    """Execute Step 3: Generate Path Sheet."""
    print_step_header(3, "Generate Path Sheet")
    
    print(f"Using dataset type: {params['dataset_type']}")
    print(f"Using base directory: {params['base_directory']}")
    print(f"Using sampling rate: {params['sampling_rate']} Hz")
    
    # Show the expected path sheet output location
    expected_output_path = params['path_sheet_path']
    print(f"\nThe path sheet will be saved to: {expected_output_path}")
    
    # Confirm before proceeding
    proceed = input("\nProceed with path sheet generation? (y/n): ").strip().lower()
    
    if proceed == 'y':
        try:
            print("Starting path sheet generation...")
            
            # Generate the path sheet using the imported function with collected parameters
            output_file = make_train_and_test_sheet(
                params['dataset_type'], 
                params['base_directory'],
                params['sampling_rate']
            )
            
            print("\nPath sheet generation completed!")
            print(f"Path sheet saved to: {expected_output_path}")
            
            return True
        except Exception as e:
            print(f"Error during path sheet generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("Path sheet generation cancelled.")
        return False


def plot_raw_signals(raw_signals, sampling_frequency, ax):
    """Plot raw signals on the given axis."""
    time = np.arange(len(raw_signals)) / sampling_frequency
    for i in range(raw_signals.shape[1]):
        ax.plot(time, raw_signals[:, i], label=f"Channel {i+1}")
    ax.set_ylabel('Amplitude')
    ax.legend()
    return ax


def execute_step_four(params, no_plots=True):
    """Execute Step 4: Preprocess Signals."""
    print_step_header(4, "Preprocess Signals")
    
    # Use the path sheet generated in the previous step
    path_sheet_path = params['path_sheet_path']
    print(f"Using path sheet: {path_sheet_path}")
    
    # Don't show plots in non-interactive mode
    show_plots = False if no_plots else input("\nDo you want to generate plots of the preprocessed signals? (y/n): ").strip().lower() == 'y'
    
    # Confirm before proceeding
    proceed = input("\nProceed with signal preprocessing? (y/n): ").strip().lower()
    
    if proceed == 'y':
        try:
            if os.path.exists(path_sheet_path):
                print("Starting signal preprocessing...")
                
                # Load the path sheet
                datasets = load_dataframe(path_sheet_path)
                print(f"Loaded {len(datasets)} dataset(s) from {path_sheet_path}")
                
                # Check required columns
                required_columns = ['file_path_raw_signals', 'sampling_frequency_in_hz', 'file_path_preprocessed_signals',
                                  'eeg1_signal_label', 'eeg2_signal_label', 'emg_signal_label']
                check_dataframe(datasets, columns=required_columns)
                
                # Define commonly used signal labels
                state_annotation_signals = ['eeg1_signal_label', 'eeg2_signal_label', 'emg_signal_label']
                
                # Set preprocessing parameters
                time_resolution = 1  # Default time resolution in seconds
                
                # Create output directory for plots if showing them
                if show_plots:
                    output_folder = os.path.join(os.path.dirname(path_sheet_path), 'output_figures')
                    os.makedirs(output_folder, exist_ok=True)
                    print(f"Created directory for figures: {output_folder}")
                
                # Process each dataset in the path sheet
                for ii, (idx, dataset) in enumerate(datasets.iterrows()):
                    print(f"\nProcessing {ii+1}/{len(datasets)}: {os.path.basename(dataset['file_path_raw_signals'])}")
                    
                    # Determine EDF signals to load
                    signal_labels = [dataset[column_name] for column_name in state_annotation_signals]
                    
                    # Load data
                    raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)
                    
                    preprocessed_signals = []
                    for signal in raw_signals.T:
                        # Fix: Convert time_resolution * sampling_frequency to integer for nperseg
                        nperseg = int(time_resolution * dataset['sampling_frequency_in_hz'])
                        
                        # Modified preprocess function call with explicit nperseg parameter
                        frequencies, time, spectrogram = get_spectrogram(
                            signal,
                            fs=dataset['sampling_frequency_in_hz'],
                            nperseg=nperseg,
                            noverlap=0
                        )
                        
                        # Process the spectrogram - filter frequency bands
                        low_cut = 1.0
                        high_cut = 90.0
                        notch_low_cut = 45.0
                        notch_high_cut = 55.0
                        
                        # Exclude ill-determined frequencies
                        mask = (frequencies >= low_cut) & (frequencies < high_cut)
                        frequencies = frequencies[mask]
                        spectrogram = spectrogram[mask]
    
                        # Exclude noise-contaminated frequencies around 50 Hz
                        mask = (frequencies >= notch_low_cut) & (frequencies <= notch_high_cut)
                        frequencies = frequencies[~mask]
                        spectrogram = spectrogram[~mask]
    
                        # Log transform and normalize
                        spectrogram = np.log(spectrogram + 1)
                        spectrogram = robust_normalize(spectrogram, p=5., axis=1, method='standard score')
                        
                        preprocessed_signals.append(spectrogram)
                    
                    # Generate visualization plots if requested
                    if show_plots:
                        # Plotting code here (omitted for brevity)
                        pass
                    
                    # Concatenate spectrograms and save
                    preprocessed_signals = np.concatenate([signal.T for signal in preprocessed_signals], axis=1)
                    export_preprocessed_signals(dataset['file_path_preprocessed_signals'], preprocessed_signals)
                    print(f"  Saved preprocessed signals to {dataset['file_path_preprocessed_signals']}")
                
                print("\nPreprocessing completed!")
                
                # Inform the user where to find the output
                preprocessed_dir = os.path.join(params['base_directory'], f"{params['dataset_type']}_set", "preprocessed_signals")
                print(f"Preprocessed signals are saved to: {preprocessed_dir}")
                
                return True
            else:
                print(f"Error: Path sheet not found at {path_sheet_path}")
                print("Please run the path sheet generation step first.")
                return False
        
        except Exception as e:
            print(f"Error during signal preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("Signal preprocessing cancelled.")
        return False


def export_intervals_with_low_confidence(file_path, state_probability, threshold=0.99, time_resolution=1):
    """Export intervals with low confidence for manual review."""
    intervals = _get_intervals(state_probability < threshold)
    scores = [len(state_probability[start:stop]) - np.sum(state_probability[start:stop]) 
              for start, stop in intervals]
    intervals = [(start * time_resolution, stop * time_resolution) for start, stop in intervals]
    notes = ['probability below threshold' for _ in intervals]
    export_review_intervals(file_path, intervals, scores, notes)


def execute_step_five(params, no_plots=True):
    """Execute Step 5: Automated Sleep Stage Annotation."""
    print_step_header(5, "Automated Sleep Stage Annotation")
    
    # Use the path sheet generated previously
    path_sheet_path = params['path_sheet_path']
    print(f"Using path sheet: {path_sheet_path}")
    
    # Ask for the trained model file path
    model_path = input("\nEnter the path to the trained model (.pickle file): ")
    
    # No plotting in non-interactive mode
    show_plots = False if no_plots else input("\nDo you want to generate plots of the annotations? (y/n): ").strip().lower() == 'y'
    
    # Confirm before proceeding
    proceed = input("\nProceed with sleep stage annotation? (y/n): ").strip().lower()
    
    if proceed == 'y':
        try:
            if os.path.exists(path_sheet_path):
                if not os.path.exists(model_path):
                    print(f"Error: Model file not found at {model_path}")
                    return False
                else:
                    print("Starting automated sleep stage annotation...")
                    
                    # Load the path sheet
                    datasets = load_dataframe(path_sheet_path)
                    print(f"Loaded {len(datasets)} dataset(s) from {path_sheet_path}")
                    
                    # Check required columns
                    required_columns = [
                        'file_path_preprocessed_signals', 
                        'file_path_automated_state_annotation', 
                        'file_path_review_intervals'
                    ]
                    if show_plots:
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
                    print(f"Loading trained model from {model_path}...")
                    annotator = StateAnnotator()
                    annotator.load(model_path)
                    
                    # Process each dataset
                    for ii, (idx, dataset) in enumerate(datasets.iterrows()):
                        print(f"\nProcessing {ii+1}/{len(datasets)}: {os.path.basename(dataset['file_path_preprocessed_signals'])}")
                        
                        # Load preprocessed signals
                        try:
                            signal_array = load_preprocessed_signals(dataset['file_path_preprocessed_signals'])
                        except Exception as e:
                            print(f"  Error loading preprocessed signals: {e}")
                            continue
                        
                        # Create output directories if needed
                        for path_key in ['file_path_automated_state_annotation', 'file_path_review_intervals']:
                            directory = os.path.dirname(dataset[path_key])
                            if not os.path.exists(directory):
                                os.makedirs(directory, exist_ok=True)
                        
                        # Predict states and probabilities
                        print("  Annotating sleep states...")
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
                        print(f"  Saved hypnogram to {dataset['file_path_automated_state_annotation']}")
                        
                        export_intervals_with_low_confidence(
                            dataset['file_path_review_intervals'],
                            state_probability,
                            threshold=0.99,
                            time_resolution=time_resolution
                        )
                        print(f"  Saved review intervals to {dataset['file_path_review_intervals']}")
                        
                        # Generate visualization if requested (omitted for brevity)
                        if show_plots:
                            # Plotting code here (omitted)
                            pass
                    
                    print("\nSleep stage annotation completed!")
                    
                    # Inform the user where to find the output
                    annotations_dir = os.path.join(params['base_directory'], f"{params['dataset_type']}_set", "automated_annotation")
                    review_dir = os.path.join(params['base_directory'], f"{params['dataset_type']}_set", "int")
                    print(f"Automated annotations are saved to: {annotations_dir}")
                    print(f"Review intervals are saved to: {review_dir}")
                    
                    return True
            else:
                print(f"Error: Path sheet not found at {path_sheet_path}")
                print("Please run the path sheet generation step first.")
                return False
        
        except Exception as e:
            print(f"Error during sleep stage annotation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("Sleep stage annotation cancelled.")
        return False


def analyze_sleep_state_proportions(annotation_file, start_time_sec, end_time_sec):
    """
    Analyze the proportion of different sleep states within a specified time window.
    
    Parameters:
    -----------
    annotation_file : str
        Path to the annotation file in Visbrain format
    start_time_sec : float
        Start time of the analysis window in seconds
    end_time_sec : float
        End time of the analysis window in seconds
        
    Returns:
    --------
    dict
        Dictionary containing the proportion of each sleep state and total duration
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
    
    return state_proportions


def batch_analyze_sleep_proportions(file_list, time_windows, output_file=None):
    """
    Analyze sleep state proportions for multiple files and time windows.
    
    Parameters:
    -----------
    file_list : list of str
        List of paths to annotation files
    time_windows : list of tuples
        List of (start_time, end_time) tuples in seconds
    output_file : str, optional
        Path to save the results as CSV. If None, results are not saved.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the results
    """
    results = []
    
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        
        for start_time, end_time in time_windows:
            try:
                # Skip invalid time windows
                if start_time >= end_time:
                    print(f"Skipping invalid time window: {start_time}-{end_time}")
                    continue
                
                print(f"Processing {file_name}: {start_time}s to {end_time}s")
                proportions = analyze_sleep_state_proportions(file_path, start_time, end_time)
                
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
                print(f"Error processing {file_name}, window {start_time}-{end_time}: {str(e)}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV if requested
    if output_file and len(results) > 0:
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return results_df


def execute_step_six(params, no_plots=True):
    """Execute Step 6: Calculate Sleep State Proportions."""
    print_step_header(6, "Calculate Sleep State Proportions")
    
    print("\nThis function analyzes the proportion of sleep states in a specified time window.")
    
    # First, suggest looking in the automated annotation directory
    suggested_annotation_dir = os.path.join(params['base_directory'], 
                                          f"{params['dataset_type']}_set", 
                                          "automated_annotation")
    
    if os.path.exists(suggested_annotation_dir):
        print(f"\nTip: Annotation files are typically located in: {suggested_annotation_dir}")
        print("These are created during Step 5 (Automated Sleep Stage Annotation).")
    
    annotation_file = input("Enter the path to the annotation file: ")
    
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found at {annotation_file}")
        return False
    
    # Read the file to get the total duration
    with open(annotation_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('*Duration_sec'):
            total_duration = float(first_line.split()[1])
            print(f"Total recording duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        else:
            print("Could not determine recording duration. Proceeding anyway.")
            total_duration = None
    
    try:
        # If start_time and end_time not provided, ask for them
        if start_time is None:
            start_time = float(input(f"Enter start time in seconds: "))
        
        if end_time is None:
            end_time = float(input(f"Enter end time in seconds{f' (max {total_duration})' if total_duration else ''}: "))
        
        if total_duration and end_time > total_duration:
            print(f"Warning: End time exceeds recording duration. Setting to maximum ({total_duration})")
            end_time = total_duration
        
        if start_time >= end_time:
            print("Error: Start time must be less than end time.")
            return False
        else:
            # Calculate proportions
            results = analyze_sleep_state_proportions(annotation_file, start_time, end_time)
            
            # Display results
            print("\nSleep State Analysis Results:")
            print("-" * 40)
            print(f"Analysis Window: {start_time:.2f} to {end_time:.2f} seconds")
            print(f"Window Duration: {results['window_duration']:.2f} seconds ({results['window_duration']/60:.2f} minutes)")
            print("\nState Proportions:")
            
            # Create a nice table of results
            key_states = ['awake', 'non-REM', 'REM']
            for state in key_states:
                if state in results:
                    duration = results.get(f'{state}_duration', 0)
                    proportion = results.get(state, 0) * 100
                    print(f"{state:>10}: {proportion:6.2f}% ({duration:.2f} seconds)")
                else:
                    print(f"{state:>10}: 0.00% (0.00 seconds)")
            
            # Display info about other states if present
            other_states = [s for s in results.keys() 
                           if s not in key_states and 
                              not s.endswith('_duration') and 
                              s != 'window_duration']
            if other_states:
                print("\nOther States:")
                for state in other_states:
                    duration = results.get(f'{state}_duration', 0)
                    proportion = results.get(state, 0) * 100
                    print(f"{state:>10}: {proportion:6.2f}% ({duration:.2f} seconds)")
            
            # Visualize as a pie chart if plots are enabled
            if not no_plots:
                # Plotting code here (omitted for brevity)
                pass
            
            # Save results to CSV
            output_folder = os.path.join(os.path.dirname(annotation_file), 'analysis_results')
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'sleep_proportions_{start_time:.0f}_{end_time:.0f}.csv')
            
            # Convert results to dataframe and save
            results_df = pd.DataFrame({
                'state': list(results.keys()),
                'value': list(results.values())
            })
            results_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
            
            # Ask if the user wants to perform batch processing
            proceed_batch = input("\nDo you want to perform batch processing of multiple files and time windows? (y/n): ").strip().lower()
            if proceed_batch == 'y':
                return execute_batch_analysis()
            
            return True
            
    except ValueError as e:
        print(f"Error with input values: {e}")
        return False


def execute_batch_analysis():
    """Execute batch analysis of sleep state proportions."""
    print("\nBatch Processing of Sleep State Proportions")
    print("------------------------------------------")
    print("This function analyzes sleep state proportions across multiple files and time windows.")
    
    # Get annotation files
    annotation_files = []
    print("\nEnter the paths to annotation files (press Enter on an empty line to finish):")
    while True:
        file_path = input("Enter annotation file path: ")
        if file_path == "":
            break
        if os.path.isfile(file_path):
            annotation_files.append(file_path)
        else:
            print("Invalid file path. Please enter a valid file path.")
    
    if not annotation_files:
        print("No valid annotation files provided. Batch processing cancelled.")
        return False
    else:
        # Get time windows
        time_windows = []
        print("\nNow enter time windows (start and end times in seconds):")
        while True:
            try:
                start_input = input("Enter start time in seconds (or press Enter to finish): ")
                if start_input == "":
                    break
                    
                start_time = float(start_input)
                end_time = float(input("Enter end time in seconds: "))
                
                if start_time < end_time:
                    time_windows.append((start_time, end_time))
                else:
                    print("Error: Start time must be less than end time.")
            except ValueError:
                print("Invalid input. Please enter numeric values.")
        
        if not time_windows:
            print("No valid time windows provided. Batch processing cancelled.")
            return False
        else:
            # Get output file
            output_file = input("\nEnter path to save results CSV (or press Enter to skip saving): ")
            if output_file.strip() == "":
                output_file = None
            
            # Perform batch analysis
            print(f"\nProcessing {len(annotation_files)} files with {len(time_windows)} time windows...")
            results_df = batch_analyze_sleep_proportions(annotation_files, time_windows, output_file)
            
            # Display summary
            print("\nProcessing completed!")
            if not results_df.empty:
                print("\nSummary of Results:")
                print(results_df)
            return True


def main():
    """Execute the full pipeline or a subset of steps."""
    parser = argparse.ArgumentParser(description="Somnotate Full Pipeline")
    parser.add_argument('--step', type=int, help='Run a specific step (1-6)')
    parser.add_argument('--steps', nargs='+', type=int, help='Run multiple steps (e.g., --steps 1 3 6)')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--no-plots', action='store_true', help='Disable all plotting')
    args = parser.parse_args()
    
    # Print header before collecting parameters
    print("\nSOMNOTATE FULL PIPELINE")
    print("======================\n")
    
    # Collect parameters used throughout the pipeline
    params = collect_common_parameters()
    no_plots = args.no_plots
    
    # Determine which steps to run
    steps_to_run = []
    if args.all:
        steps_to_run = list(range(1, 7))
    elif args.step:
        steps_to_run = [args.step]
    elif args.steps:
        steps_to_run = sorted(args.steps)
    else:
        print("\nWhich steps would you like to run?")
        print("  1. MAT to CSV Conversion")
        print("  2. Generate EDF and Visbrain Format Files")
        print("  3. Generate Path Sheet")
        print("  4. Preprocess Signals")
        print("  5. Automated Sleep Stage Annotation")
        print("  6. Calculate Sleep State Proportions")
        print("  0. Run all steps")
        
        choice = input("\nEnter step numbers separated by spaces (e.g., 1 3 5), or 0 for all steps: ")
        if choice.strip() == "0":
            steps_to_run = list(range(1, 7))
        else:
            try:
                steps_to_run = sorted([int(s) for s in choice.split()])
            except ValueError:
                print("Invalid input. Exiting.")
                return
    
    # Validate steps
    valid_steps = list(range(1, 7))
    if not all(step in valid_steps for step in steps_to_run):
        print(f"Error: Some steps are invalid. Valid steps are: {valid_steps}")
        return
    
    # Execute selected steps
    print(f"\nExecuting steps: {steps_to_run}")
    
    step_functions = {
        1: lambda p: execute_step_one(p),
        2: lambda p: execute_step_two(p),
        3: lambda p: execute_step_three(p),
        4: lambda p: execute_step_four(p, no_plots=no_plots),
        5: lambda p: execute_step_five(p, no_plots=no_plots),
        6: lambda p: execute_step_six(p, no_plots=no_plots)
    }
    
    step_results = {}
    for step in steps_to_run:
        step_results[step] = step_functions[step](params)
        print()  # Add some space between steps
    
    # Print summary
    print("\nPipeline Execution Summary:")
    print("--------------------------")
    for step in steps_to_run:
        status = "Completed" if step_results.get(step, False) else "Failed"
        step_name = {
            1: "MAT to CSV Conversion",
            2: "Generate EDF and Visbrain Format Files",
            3: "Generate Path Sheet",
            4: "Preprocess Signals",
            5: "Automated Sleep Stage Annotation",
            6: "Calculate Sleep State Proportions"
        }[step]
        print(f"Step {step} ({step_name}): {status}")
    
    print("\nPipeline execution completed!")


if __name__ == "__main__":
    main()
