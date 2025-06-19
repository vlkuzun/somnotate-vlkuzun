"""
Convert MAT files to CSV format, extracting EEG/EMG data and organizing it for the Somnotate pipeline.
"""

import os
import numpy as np
import pandas as pd
import h5py


def mat_to_csv(file_paths, output_directory_path, sampling_rate, sleep_stage_resolution=10):
    """
    Convert MAT files to CSV format.
    
    Parameters:
    -----------
    file_paths : list of str
        Paths to the MAT files to convert
    output_directory_path : str
        Path to the directory where CSV files will be saved
    sampling_rate : float
        Sampling rate in Hz
    sleep_stage_resolution : int, optional
        Time resolution of sleep stage annotations in seconds
        
    Returns:
    --------
    list
        List of paths to the created CSV files
    """
    output_files = []
    
    for path in file_paths:
        print(f"Processing {path}...")
        
        # Extract the base filename without extension
        base_filename = os.path.splitext(os.path.basename(path))[0]
        output_file = os.path.join(output_directory_path, f"{base_filename}.csv")
        
        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping.")
            output_files.append(output_file)
            continue
            
        # Load MAT file
        try:
            with h5py.File(path, 'r') as f:
                # Extract EEG and EMG signals
                # Assuming a specific structure - modify according to your MAT file structure
                eeg1_data = np.array(f.get('eeg1'))
                eeg2_data = np.array(f.get('eeg2'))
                emg_data = np.array(f.get('emg'))
                
                # Create a DataFrame
                df = pd.DataFrame({
                    'EEG1': eeg1_data.flatten(),
                    'EEG2': eeg2_data.flatten(),
                    'EMG': emg_data.flatten()
                })
                
                # Initialize sleep stage column with NaN
                df['sleepStage'] = np.nan
                
                # Save to CSV
                df.to_csv(output_file, index=False)
                print(f"Saved to {output_file}")
                output_files.append(output_file)
        
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    return output_files


if __name__ == "__main__":
    # Ask for inputs only when run as a script, not when imported
    print("MAT to CSV Conversion")
    print("-" * 20)
    
    # Get input parameters
    file_paths = []
    print("\nEnter the full paths of .mat files to convert (press Enter on an empty line to finish):")
    while True:
        file_path = input("Enter file path: ")
        if file_path == "":
            break
        if os.path.isfile(file_path) and file_path.endswith('.mat'):
            file_paths.append(file_path)
        else:
            print("Invalid file path. Please enter a valid .mat file path.")
            
    output_directory_path = input("Enter output directory path: ")
    sampling_rate = float(input("Enter sampling rate (e.g., 512.0): "))
    sleep_stage_resolution = int(input("Enter sleep stage resolution in seconds (e.g., 10): "))
    
    # Run the conversion
    if file_paths and output_directory_path:
        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)
            
        mat_to_csv(file_paths, output_directory_path, sampling_rate, sleep_stage_resolution)
    else:
        print("Missing required inputs.")
