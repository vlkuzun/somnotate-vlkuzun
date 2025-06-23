#!/usr/bin/env python

"""
Interactive Somnotate Pipeline

This script provides an interactive command-line interface for running the complete somnotate pipeline:
1. Convert MAT files to CSV format
2. Generate EDF and Visbrain format files for visualization and analysis
3. Generate path sheet for organizing files
4. Preprocess signals for sleep stage analysis
5. Apply automated sleep stage annotation
6. Analyze sleep state proportions

Run the script and follow the prompts to execute each step in sequence.
"""

import os
import sys
import re  # Add this import for regex pattern matching
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyedflib
from functools import partial

# Add the somnotate pipeline directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import from somnotate pipeline
try:
    from mat_to_csv import mat_to_csv
    from make_path_sheet import make_train_and_test_sheet
    from preprocess_signals import preprocess
    from data_io import load_dataframe, check_dataframe, load_raw_signals, export_preprocessed_signals
    from data_io import load_preprocessed_signals, export_hypnogram, export_review_intervals
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure you've installed all required packages and are running from the correct directory.")
    sys.exit(1)

# Import spectrogram function
try:
    from lspopt import spectrogram_lspopt
    get_spectrogram = partial(spectrogram_lspopt, c_parameter=20.)
except ImportError:
    print("Warning: lspopt not installed. Using scipy's spectrogram instead.")
    from scipy.signal import spectrogram as get_spectrogram

# Import somnotate utilities if available
try:
    from somnotate._utils import robust_normalize, convert_state_vector_to_state_intervals, _get_intervals
    from somnotate._automated_state_annotation import StateAnnotator
except ImportError:
    print("Warning: somnotate module not available. Some functionality may be limited.")
    
    # Define fallback function for robust_normalize
    def robust_normalize(x, p=10.0, axis=0, method='standard score'):
        """Normalize array x by subtracting the median and dividing by IQR."""
        med = np.median(x, axis=axis, keepdims=True)
        iqr = np.percentile(x, 50 + p/2, axis=axis, keepdims=True) - np.percentile(x, 50 - p/2, axis=axis, keepdims=True)
        if method == 'standard score':
            return (x - med) / (iqr + 1e-8)
        elif method == 'min max':
            return (x - med) / (iqr + 1e-8) * 2 + 0.5
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def print_header(title):
    """Print a formatted header for each section."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def get_user_input(prompt_text, options=None, is_path=False, is_multiple=False, 
                  is_float=False, is_int=False, allow_empty=False):
    """
    Helper function to get user input with validation.
    
    Parameters:
    -----------
    prompt_text : str
        Text to display as prompt
    options : list, optional
        List of valid options
    is_path : bool, optional
        Whether the input is a file path
    is_multiple : bool, optional
        Whether to accept multiple values (space-separated)
    is_float : bool, optional
        Whether to convert input to float
    is_int : bool, optional
        Whether to convert input to int
    allow_empty : bool, optional
        Whether to allow empty input
    
    Returns:
    --------
    str, list, float, or int
        The validated user input
    """
    while True:
        if is_multiple:
            print(f"{prompt_text} (space-separated values):")
            value = input().strip()
            if not value and not allow_empty:
                print("Input cannot be empty. Please try again.")
                continue
            if value or not allow_empty:
                values = value.split()
                return values
            return []
        else:
            print(f"{prompt_text}:")
            value = input().strip()
            
            if not value and not allow_empty:
                print("Input cannot be empty. Please try again.")
                continue
            
            if not value and allow_empty:
                return "" if not (is_float or is_int) else None
            
            if is_path and value:
                value = os.path.abspath(os.path.expanduser(value))
                if not os.path.exists(value) and not prompt_text.startswith(("Output", "Save")):
                    print(f"Warning: Path '{value}' does not exist.")
                    continue_anyway = input("Continue anyway? (y/n): ").lower().startswith('y')
                    if not continue_anyway:
                        continue
            
            if is_float:
                try:
                    return float(value)
                except ValueError:
                    print("Please enter a valid number.")
                    continue
                    
            if is_int:
                try:
                    return int(value)
                except ValueError:
                    print("Please enter a valid integer.")
                    continue
                    
            if options and value not in options:
                print(f"Please choose from: {', '.join(options)}")
                continue
                
            return value


def collect_common_parameters():
    """Collect common input parameters that will be reused throughout the pipeline."""
    print_header("Common Input Parameters")
    print("Enter the common parameters that will be used throughout the pipeline:")
    
    # Dataset type
    dataset_type = get_user_input("Enter dataset type ('train', 'test', or 'to_score')", 
                               options=['train', 'test', 'to_score'])
    
    # Base directory
    base_directory = get_user_input("Enter the base somnotate directory path")
    
    # Sampling rate
    sampling_rate = get_user_input("Enter the sampling rate in Hz (e.g., 512.0)", is_float=True)
    
    # Sleep stage resolution
    sleep_stage_resolution = get_user_input("Enter the sleep stage resolution in seconds (e.g., 10)", is_int=True)
    
    # Mouse, session, recording information
    print("\nEnter subject and recording information:")
    mouse_ids = get_user_input("Enter mouse IDs (e.g., sub-001 sub-002)", is_multiple=True)
    sessions = get_user_input("Enter session IDs (e.g., ses-01 ses-02)", is_multiple=True)
    recordings = get_user_input("Enter recording IDs (e.g., recording-01 recording-02)", is_multiple=True)
    extra_info = get_user_input("Enter any extra details about the recording (leave blank if not applicable)", allow_empty=True)
    
    # Define derived paths for reuse
    output_directory_path = os.path.join(base_directory, f"{dataset_type}_set", f"{dataset_type}_csv_files")
    path_sheet_path = os.path.join(base_directory, f"{dataset_type}_set", f"{dataset_type}_sheet.csv")
    
    # Print summary of all inputs
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
        'path_sheet_path': path_sheet_path
    }


def generate_edf_and_visbrain_formats(mouse_ids, sessions, recordings, extra_info, test_train_or_to_score, base_directory, sample_frequency):
    """
    Generate EDF and Visbrain stage duration format files from CSV files.
    
    Parameters:
    -----------
    mouse_ids : list of str
        Mouse IDs
    sessions : list of str
        Session IDs
    recordings : list of str
        Recording IDs
    extra_info : str
        Additional information for file names
    test_train_or_to_score : str
        Dataset type ('train', 'test', or 'to_score')
    base_directory : str
        Base directory path
    sample_frequency : float
        Sampling frequency in Hz
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Define output directories
        csv_input_dir = os.path.join(base_directory, f"{test_train_or_to_score}_set/{test_train_or_to_score}_csv_files")
        edf_output_dir = os.path.join(base_directory, f"{test_train_or_to_score}_set", 'edfs')
        annotations_output_dir = os.path.join(base_directory, f"{test_train_or_to_score}_set", 
                                            f"{test_train_or_to_score}_manual_annotation")
        
        if not os.path.exists(edf_output_dir):
            os.makedirs(edf_output_dir)
        if not os.path.exists(annotations_output_dir):
            os.makedirs(annotations_output_dir)

        # Process each CSV file and generate output
        for mouse_id in mouse_ids:
            for session in sessions:
                for recording in recordings:
                    # Prepare filenames
                    base_filename = f"{mouse_id}_{session}_{recording}"
                    if extra_info:
                        csv_file = os.path.join(csv_input_dir, f"{base_filename}_{extra_info}.csv")
                        edf_file = os.path.join(edf_output_dir, f"output_{base_filename}_{extra_info}.edf")
                        visbrain_file = os.path.join(annotations_output_dir, f"annotations_visbrain_{base_filename}_{extra_info}.txt")
                    else:
                        csv_file = os.path.join(csv_input_dir, f"{base_filename}.csv")
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

                    # Extract data
                    eeg1_data = df["EEG1"].to_numpy()
                    eeg2_data = df["EEG2"].to_numpy()
                    emg_data = df["EMG"].to_numpy()
                    all_data = np.array([eeg1_data, eeg2_data, emg_data])

                    # Create EDF file
                    f = pyedflib.EdfWriter(edf_file, len(all_data), file_type=pyedflib.FILETYPE_EDFPLUS)

                    # Define EDF header
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

                    # Write data and close
                    f.writeSamples(all_data)
                    f.close()

                    # Prepare Visbrain annotations
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

                    # Write annotations
                    last_time_value = annotations[-1][1]
                    with open(visbrain_file, "w") as f:
                        f.write(f"*Duration_sec    {last_time_value}\n")
                        f.write("*Datafile\tUnspecified\n")
                        for start, end, stage in annotations:
                            stage_label = {1: "awake", 2: "non-REM", 3: "REM", 4: 'ambiguous', 5: 'doubt'}.get(stage, "Undefined")
                            f.write(f"{stage_label}    {end}\n")

                    print(f"EDF and annotations created for {mouse_id}, {session}, {recording}")
                    
        return True
    except Exception as e:
        print(f"Error generating EDF and Visbrain formats: {e}")
        return False


def convert_mat_to_csv(params):
    """Convert MAT files to CSV format."""
    print_header("MAT to CSV Conversion")
    
    # Get file paths
    file_paths = []
    print("Enter the full paths of .mat files to convert (press Enter on an empty line to finish):")
    while True:
        file_path = input("Enter file path (or Enter to finish): ").strip()
        if not file_path:
            break
        if os.path.isfile(file_path) and file_path.endswith('.mat'):
            file_paths.append(file_path)
        else:
            print("Invalid file path. Please enter a valid .mat file path.")
    
    if not file_paths:
        print("No files selected. Skipping MAT to CSV conversion.")
        return False
    
    # Print summary
    print(f"\nFiles to process: {len(file_paths)}")
    for i, path in enumerate(file_paths):
        print(f"  {i+1}. {path}")
    
    # Confirm and execute
    proceed = input("\nProceed with conversion? (y/n): ").strip().lower().startswith('y')
    if proceed:
        try:
            # Create output directory
            if not os.path.exists(params['output_directory_path']):
                os.makedirs(params['output_directory_path'])
                print(f"Created output directory: {params['output_directory_path']}")
            
            # Execute conversion
            mat_to_csv(file_paths, params['output_directory_path'], 
                     params['sampling_rate'], params['sleep_stage_resolution'])
            
            # List created files
            output_files = [f for f in os.listdir(params['output_directory_path']) if f.endswith('.csv')]
            print(f"\nCreated {len(output_files)} CSV files in {params['output_directory_path']}")
            return True
        
        except Exception as e:
            print(f"Error during conversion: {e}")
            return False
    else:
        print("MAT to CSV conversion cancelled.")
        return False


def generate_edf_files(params):
    """Generate EDF and Visbrain format files."""
    print_header("EDF and Visbrain Format Generation")
    
    # Calculate output paths
    edf_output_dir = os.path.join(params['base_directory'], f"{params['dataset_type']}_set", 'edfs')
    annotations_output_dir = os.path.join(params['base_directory'], f"{params['dataset_type']}_set", 
                                        f"{params['dataset_type']}_manual_annotation")
    
    print(f"EDF files will be saved to: {edf_output_dir}")
    print(f"Annotation files will be saved to: {annotations_output_dir}")
    
    # Confirm and execute
    proceed = input("\nProceed with EDF generation? (y/n): ").strip().lower().startswith('y')
    if proceed:
        success = generate_edf_and_visbrain_formats(
            params['mouse_ids'],
            params['sessions'],
            params['recordings'],
            params['extra_info'],
            params['dataset_type'],
            params['base_directory'],
            params['sampling_rate']
        )
        return success
    else:
        print("EDF generation cancelled.")
        return False


def generate_path_sheet(params):
    """Generate path sheet for organizing files."""
    print_header("Path Sheet Generation")
    
    # Show expected output path
    expected_output_path = params['path_sheet_path']
    print(f"The path sheet will be saved to: {expected_output_path}")
    
    # Confirm and execute
    proceed = input("\nProceed with path sheet generation? (y/n): ").strip().lower().startswith('y')
    if proceed:
        try:
            make_train_and_test_sheet(params['dataset_type'], params['base_directory'], params['sampling_rate'])
            
            if os.path.exists(expected_output_path):
                print(f"Path sheet successfully created at {expected_output_path}")
                return True
            else:
                print("Path sheet generation completed but the file was not found at the expected location.")
                return False
        
        except Exception as e:
            print(f"Error during path sheet generation: {e}")
            return False
    else:
        print("Path sheet generation cancelled.")
        return False


def preprocess_signals(params):
    """Preprocess signals for sleep stage analysis."""
    print_header("Signal Preprocessing")
    
    # Check if path sheet exists
    if not os.path.exists(params['path_sheet_path']):
        print(f"Path sheet not found at {params['path_sheet_path']}")
        print("Please run the path sheet generation step first.")
        return False
    
    # Ask about visualization
    show_plots = input("Do you want to generate plots of preprocessed signals? (y/n): ").strip().lower().startswith('y')
    
    # Confirm and execute
    proceed = input("\nProceed with signal preprocessing? (y/n): ").strip().lower().startswith('y')
    if not proceed:
        print("Signal preprocessing cancelled.")
        return False
    
    try:
        print("Loading path sheet...")
        datasets = load_dataframe(params['path_sheet_path'])
        print(f"Loaded {len(datasets)} dataset(s)")
        
        # Check required columns
        required_columns = ['file_path_raw_signals', 'sampling_frequency_in_hz', 'file_path_preprocessed_signals',
                          'eeg1_signal_label', 'eeg2_signal_label', 'emg_signal_label']
        check_dataframe(datasets, columns=required_columns)
        
        # Set preprocessing parameters
        time_resolution = 1  # Default time resolution in seconds
        state_annotation_signals = ['eeg1_signal_label', 'eeg2_signal_label', 'emg_signal_label']
        
        # Create output directory for plots
        if show_plots:
            output_folder = os.path.join(os.path.dirname(params['path_sheet_path']), 'output_figures')
            os.makedirs(output_folder, exist_ok=True)
        
        # Define helper function for plotting
        def plot_raw_signals(raw_signals, sampling_frequency, ax):
            time = np.arange(len(raw_signals)) / sampling_frequency
            for i in range(raw_signals.shape[1]):
                ax.plot(time, raw_signals[:, i], label=f"Channel {i+1}")
            ax.set_ylabel('Amplitude')
            ax.legend()
            return ax
        
        # Process each dataset
        print("\nStarting preprocessing...")
        for ii, (idx, dataset) in enumerate(datasets.iterrows()):
            print(f"Processing {ii+1}/{len(datasets)}: {os.path.basename(dataset['file_path_raw_signals'])}")
            
            # Get signal labels and load data
            signal_labels = [dataset[column_name] for column_name in state_annotation_signals]
            raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)
            
            # Create output directory if needed
            output_dir = os.path.dirname(dataset['file_path_preprocessed_signals'])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Process each signal
            preprocessed_signals = []
            for signal in raw_signals.T:
                # Convert time_resolution * sampling_frequency to integer for nperseg
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
            
            # Visualization
            if show_plots:
                fig, axes = plt.subplots(1+len(preprocessed_signals), 1, figsize=(12, 8), sharex=True)
                
                # Plot raw signals
                plot_raw_signals(
                    raw_signals,
                    sampling_frequency=dataset['sampling_frequency_in_hz'],
                    ax=axes[0] if isinstance(axes, np.ndarray) else axes
                )
                
                # Plot spectrograms
                for i, signal in enumerate(preprocessed_signals):
                    if isinstance(axes, np.ndarray):
                        ax = axes[i+1]
                    else:
                        fig_new, ax = plt.subplots(figsize=(12, 4))
                    
                    im = ax.imshow(signal, 
                                  aspect='auto', 
                                  origin='lower', 
                                  extent=[time[0], time[-1], frequencies[0], frequencies[-1]],
                                  cmap='viridis')
                    ax.set_ylabel('Frequency (Hz)')
                    plt.colorbar(im, ax=ax, label='Power (normalized)')
                
                # Set common labels
                if isinstance(axes, np.ndarray):
                    axes[-1].set_xlabel('Time (s)')
                else:
                    axes.set_xlabel('Time (s)')
                
                # Save figure
                plt.tight_layout()
                fig_path = os.path.join(output_folder, f'preprocessed_signals_{ii}.png')
                plt.savefig(fig_path)
                plt.close(fig)
            
            # Save preprocessed signals
            preprocessed_signals = np.concatenate([signal.T for signal in preprocessed_signals], axis=1)
            export_preprocessed_signals(dataset['file_path_preprocessed_signals'], preprocessed_signals)
            print(f"  Saved to {dataset['file_path_preprocessed_signals']}")
        
        print("\nPreprocessing completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error during signal preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def annotate_sleep_stages(params):
    """Annotate sleep stages using a trained model."""
    print_header("Automated Sleep Stage Annotation")
    
    # Check for path sheet
    if not os.path.exists(params['path_sheet_path']):
        print(f"Path sheet not found at {params['path_sheet_path']}")
        print("Please run the path sheet generation step first.")
        return False
    
    # Get trained model path
    model_path = get_user_input("Enter the path to the trained model (.pickle file)", is_path=True)
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return False
    
    # Ask about visualization
    show_plots = input("Do you want to generate plots of the annotations? (y/n): ").strip().lower().startswith('y')
    
    # Confirm and execute
    proceed = input("\nProceed with sleep stage annotation? (y/n): ").strip().lower().startswith('y')
    if not proceed:
        print("Sleep stage annotation cancelled.")
        return False
    
    try:
        print("Starting automated sleep stage annotation...")
        
        # Check if required modules are available
        if 'StateAnnotator' not in globals():
            print("Error: Required annotation modules not found.")
            print("Please ensure the somnotate package is properly installed.")
            return False
        
        # Load the path sheet
        datasets = load_dataframe(params['path_sheet_path'])
        print(f"Loaded {len(datasets)} dataset(s)")
        
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
        
        # Helper function for review intervals
        def export_intervals_with_low_confidence(file_path, state_probability, threshold=0.99, time_resolution=1):
            intervals = _get_intervals(state_probability < threshold)
            scores = [len(state_probability[start:stop]) - np.sum(state_probability[start:stop]) 
                      for start, stop in intervals]
            intervals = [(start * time_resolution, stop * time_resolution) for start, stop in intervals]
            notes = ['probability below threshold' for _ in intervals]
            export_review_intervals(file_path, intervals, scores, notes)
        
        # Define state mappings and parameters
        int_to_state = {
            1: "awake",
            2: "non-REM",
            3: "REM",
            4: "ambiguous",
            5: "doubt"
        }
        time_resolution = 1
        
        # Create output directory for plots
        if show_plots:
            output_folder = os.path.join(os.path.dirname(params['path_sheet_path']), 'output_figures')
            os.makedirs(output_folder, exist_ok=True)
        
        # Load the trained model
        print(f"Loading trained model from {model_path}...")
        annotator = StateAnnotator()
        annotator.load(model_path)
        
        # Process each dataset
        for ii, (idx, dataset) in enumerate(datasets.iterrows()):
            print(f"Processing {ii+1}/{len(datasets)}: {os.path.basename(dataset['file_path_preprocessed_signals'])}")
            
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
            
            # Generate visualization if requested
            if show_plots:
                print("  Generating visualization...")
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
                fig_path = os.path.join(output_folder, f'annotation_{ii}.png')
                plt.savefig(fig_path)
                plt.close(fig)
        
        print("\nSleep stage annotation completed!")
        return True
    
    except Exception as e:
        print(f"Error during sleep stage annotation: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_sleep_state_proportions(annotation_file, start_time_sec, end_time_sec):
    """
    Analyze the proportion of different sleep states within a time window.
    
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


def analyze_sleep_stages(params):
    """Analyze sleep state proportions."""
    print_header("Sleep State Proportion Analysis")
    
    # Get annotation file
    annotation_file = get_user_input("Enter the path to the annotation file", is_path=True)
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found at {annotation_file}")
        return False
    
    # Use parameters directly from the input without pattern matching
    # Use mouse ID as subject directly
    subject_id = params['mouse_ids'][0] if params['mouse_ids'] else "sub-unknown"
    session_id = params['sessions'][0] if params['sessions'] else "ses-unknown"
    recording_id = params['recordings'][0] if params['recordings'] else "recording-unknown"
    time_info = params['extra_info'] if params['extra_info'] else "time-0-24h"
    
    # Get time window
    try:
        # Read the file to get the total duration
        with open(annotation_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('*Duration_sec'):
                total_duration_sec = float(first_line.split()[1])
                total_duration_min = total_duration_sec / 60.0
                print(f"Total recording duration: {total_duration_min:.2f} minutes ({total_duration_sec:.2f} seconds)")
            else:
                print("Could not determine recording duration. Proceeding anyway.")
                total_duration_sec = None
                total_duration_min = None
        
        # Get time window in minutes
        start_time_min = get_user_input("Enter start time in minutes", is_float=True)
        end_time_min = get_user_input(f"Enter end time in minutes{f' (max {total_duration_min:.2f})' if total_duration_min else ''}", 
                                is_float=True)
        
        # Convert minutes to seconds for internal processing
        start_time_sec = start_time_min * 60.0
        end_time_sec = end_time_min * 60.0
        
        if total_duration_sec and end_time_sec > total_duration_sec:
            print(f"Warning: End time exceeds recording duration. Setting to maximum ({total_duration_min:.2f} minutes)")
            end_time_sec = total_duration_sec
            end_time_min = total_duration_min
        
        if start_time_sec >= end_time_sec:
            print("Error: Start time must be less than end time.")
            return False
        
        # Calculate proportions
        print("Analyzing sleep state proportions...")
        results = analyze_sleep_state_proportions(annotation_file, start_time_sec, end_time_sec)
        
        # Convert durations from seconds to minutes for display
        window_duration_min = results['window_duration'] / 60.0
        
        # Display results
        print("\nSleep State Analysis Results:")
        print("-" * 40)
        print(f"Analysis Window: {start_time_min:.2f} to {end_time_min:.2f} minutes")
        print(f"Window Duration: {window_duration_min:.2f} minutes ({results['window_duration']:.2f} seconds)")
        print("\nState Proportions:")
        
        # Create a table of results
        key_states = ['awake', 'non-REM', 'REM']
        for state in key_states:
            if state in results:
                duration_sec = results.get(f'{state}_duration', 0)
                duration_min = duration_sec / 60.0
                proportion = results.get(state, 0) * 100
                print(f"{state:>10}: {proportion:6.2f}% ({duration_min:.2f} minutes)")
            else:
                print(f"{state:>10}: 0.00% (0.00 minutes)")
        
        # Display info about other states if present
        other_states = [s for s in results.keys() 
                       if s not in key_states and 
                          not s.endswith('_duration') and 
                          s != 'window_duration']
        if other_states:
            print("\nOther States:")
            for state in other_states:
                duration_sec = results.get(f'{state}_duration', 0)
                duration_min = duration_sec / 60.0
                proportion = results.get(state, 0) * 100
                print(f"{state:>10}: {proportion:6.2f}% ({duration_min:.2f} minutes)")
        
        # Visualize as a pie chart
        labels = []
        sizes = []
        for state in results:
            if not state.endswith('_duration') and state != 'window_duration':
                labels.append(state)
                sizes.append(results[state])
        
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
        centre_circle = plt.Circle((0,0), 0.60, fc='white')
        fig.gca().add_artist(centre_circle)
        
        # Format the title according to the specified structure
        subject_id = f"{subject_id}" if subject_id else "sub-unknown"
        session_id = f"{session_id}" if session_id else "ses-unknown"
        recording_id = f"{recording_id}" if recording_id else "recording-unknown"
        time_info = f"{time_info}" if time_info else "time-0-24h"  # Use extra_info as time range or default
        
        # Create title with the specified format using direct input parameters
        title = f"{subject_id} {session_id} {recording_id} {time_info} Sleep State Proportions ({start_time_min:.1f}-{end_time_min:.1f} min)"
        ax.set_title(title, size=14)
        plt.text(0, 0, f"Total: {window_duration_min:.2f} min", ha='center', va='center', fontsize=12)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        plt.tight_layout()
        
        # Save the figure with the specified filename format using direct input parameters
        output_folder = os.path.join(os.path.dirname(annotation_file), 'output_figures')
        os.makedirs(output_folder, exist_ok=True)
        
        # Create filename structure directly from input parameters
        fig_path = os.path.join(output_folder, 
                              f'sleep_proportions_{subject_id}_{session_id}_{recording_id}_{time_info}_{start_time_min:.1f}_{end_time_min:.1f}_min.png')
        plt.savefig(fig_path)
        print(f"\nPie chart saved to {fig_path}")
        plt.show()
        
        # Ask if user wants to save results to CSV
        save_csv = input("\nDo you want to save these results to CSV? (y/n): ").strip().lower().startswith('y')
        if save_csv:
            output_file = get_user_input("Enter output CSV path", is_path=True)
            # Convert to DataFrame for saving with all metadata
            df = pd.DataFrame([{
                'file': os.path.basename(annotation_file),
                'subject': subject_id or "",
                'session': session_id or "",
                'recording': recording_id or "",
                'extra_info': time_info or "",
                'start_time_min': start_time_min,
                'end_time_min': end_time_min,
                'window_duration_min': window_duration_min,
                **{k: v for k, v in results.items() if not k.endswith('_duration') and k != 'window_duration'}
            }])
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return True
    
    except Exception as e:
        print(f"Error during sleep state analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


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


def batch_analysis(params):
    """Perform batch analysis of sleep state proportions."""
    print_header("Batch Sleep State Proportion Analysis")
    
    # Get annotation files
    annotation_files = []
    print("Enter the paths to annotation files (press Enter on an empty line to finish):")
    while True:
        file_path = input("Enter annotation file path (or Enter to finish): ").strip()
        if not file_path:
            break
        if os.path.isfile(file_path):
            annotation_files.append(file_path)
        else:
            print("Invalid file path. Please enter a valid file path.")
    
    if not annotation_files:
        print("No valid annotation files provided. Batch processing cancelled.")
        return False
    
    # Get time windows
    time_windows = []
    print("\nNow enter time windows (start and end times in seconds):")
    while True:
        try:
            start_input = input("Enter start time in seconds (or press Enter to finish): ").strip()
            if not start_input:
                break
                
            start_time = float(start_input)
            end_time = float(input("Enter end time in seconds: ").strip())
            
            if start_time < end_time:
                time_windows.append((start_time, end_time))
            else:
                print("Error: Start time must be less than end time.")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    
    if not time_windows:
        print("No valid time windows provided. Batch processing cancelled.")
        return False
    
    # Get output file
    output_file = get_user_input("Enter path to save results CSV", is_path=True)
    
    # Execute batch analysis
    try:
        print(f"\nProcessing {len(annotation_files)} files with {len(time_windows)} time windows...")
        results_df = batch_analyze_sleep_proportions(annotation_files, time_windows, output_file)
        
        if not results_df.empty:
            print("\nProcessing completed!")
            print("\nSummary of Results:")
            # Print a summary of the results
            print(f"Total analyses performed: {len(results_df)}")
            print("Average proportions across all files and windows:")
            
            # Get the state columns
            state_columns = [col for col in results_df.columns if col.endswith('_proportion')]
            for col in state_columns:
                state = col.replace('_proportion', '')
                mean_prop = results_df[col].mean() * 100
                print(f"  {state:>10}: {mean_prop:6.2f}%")
            
            return True
        else:
            print("No results were generated.")
            return False
    
    except Exception as e:
        print(f"Error during batch analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline():
    """Main function to run the interactive somnotate pipeline."""
    print_header("Interactive Somnotate Pipeline")
    print("Welcome to the Interactive Somnotate Pipeline!")
    print("\nThis script will guide you through the complete sleep analysis process.")
    print("Follow the prompts to execute each step of the pipeline.\n")
    
    # Collect common parameters
    params = collect_common_parameters()
    
    # Execute pipeline steps in sequence
    steps = [
        ("Convert MAT files to CSV", convert_mat_to_csv),
        ("Generate EDF and Visbrain format files", generate_edf_files),
        ("Generate path sheet", generate_path_sheet),
        ("Preprocess signals", preprocess_signals),
        ("Annotate sleep stages", annotate_sleep_stages),
        ("Analyze sleep stages", analyze_sleep_stages),
        ("Batch analysis", batch_analysis)
    ]
    
    step_results = {}
    
    for i, (step_name, step_func) in enumerate(steps):
        print_header(f"Step {i+1}: {step_name}")
        
        # Ask if user wants to run this step
        run_step = input(f"Do you want to run this step? (y/n): ").strip().lower().startswith('y')
        if not run_step:
            print(f"Skipping step {i+1}: {step_name}")
            continue
        
        # For steps that depend on previous steps, check if they were successful
        if i >= 3 and step_name != "Analyze sleep stages" and step_name != "Batch analysis":  # Signal processing and annotation steps
            if not step_results.get("Generate path sheet", False):
                print(f"Cannot run {step_name} because path sheet generation was not completed.")
                print("Please run the path sheet generation step first.")
                continue
        
        # Run the step
        success = step_func(params)
        step_results[step_name] = success
        
        if success:
            print(f"\nStep {i+1}: {step_name} completed successfully!")
        else:
            print(f"\nStep {i+1}: {step_name} did not complete successfully.")
    
    print_header("Pipeline Complete")
    print("Summary of completed steps:")
    for step_name, success in step_results.items():
        status = "✓ Completed" if success else "✗ Failed"
        print(f"- {step_name}: {status}")
    
    print("\nThank you for using the Interactive Somnotate Pipeline!")


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("The pipeline has encountered an error and cannot continue.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("The pipeline has encountered an error and cannot continue.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("The pipeline has encountered an error and cannot continue.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("The pipeline has encountered an error and cannot continue.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("The pipeline has encountered an error and cannot continue.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("The pipeline has encountered an error and cannot continue.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

