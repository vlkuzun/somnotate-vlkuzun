import numpy as np
import pandas as pd
import h5py
import os

def mat_to_csv(file_paths, output_directory_path, sampling_rate, sleep_stage_resolution):
    '''
    Converts .mat files extracted from Spike2 into .csv files to be used in the somnotate pipeline.
    Ensures the length of upsampled sleep stages and EEG data match, truncating or padding if necessary.
    Inputs:
    file_paths: list of str, paths to the selected .mat files
    output_directory_path: str, path to the directory where the .csv files should be saved
    sampling_rate: int, Hz (samples per second)
    sleep_stage_resolution: int, seconds
    '''
    for file_path in file_paths:
        # Extract the base filename without the extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        with h5py.File(file_path, 'r') as raw_data:
            print(f'Processing file: {file_path}')

            # Initialize variables to store data
            eeg1_data, eeg2_data, emg_data, sleep_stages = None, None, None, None

            # Iterate over all keys in the HDF5 file to extract data
            for key in raw_data.keys():
                if key.endswith('_EEG_EEG1A_B') or key.endswith('_EEGorig'):
                    eeg1_data = np.array(raw_data[key]['values'])
                elif key.endswith('_EEG_EEG2A_B') or key.endswith('_EEGorig'):
                    eeg2_data = np.array(raw_data[key]['values'])
                elif key.endswith('_EMG_EMG'):
                    emg_data = np.array(raw_data[key]['values'])
                elif key.endswith('_Stage_1_'):
                    sleep_stages = np.array(raw_data[key]['codes'])
                    sleep_stages = sleep_stages[0, :]
            
            # Check if data was found
            if eeg1_data is not None:
                print("EEG1 data extracted successfully.")
            if eeg2_data is not None:
                print("EEG2 data extracted successfully.")
            if emg_data is not None:
                print("EMG data extracted successfully.")
            if sleep_stages is not None:
                print("Sleep stage data extracted successfully.")

            # Format data for saving to a CSV file
            eeg1_flattened = eeg1_data.flatten()
            eeg2_flattened = eeg2_data.flatten()
            emg_flattened = emg_data.flatten()
            assert eeg1_flattened.shape == eeg2_flattened.shape == emg_flattened.shape, "The flattened shapes of the EEG and EMG data do not match"

            # Upsample sleep stages to match EEG/EMG data resolution
            upsampled_sleep_stages = np.repeat(sleep_stages, sampling_rate * sleep_stage_resolution)
            if len(upsampled_sleep_stages) != len(eeg1_flattened):
                print(f"Length of upsampled sleep stages ({len(upsampled_sleep_stages)}) does not match length of EEG data ({len(eeg1_flattened)})")
                if len(upsampled_sleep_stages) > len(eeg1_flattened):
                    upsampled_sleep_stages = upsampled_sleep_stages[:len(eeg1_flattened)]
                else:
                    padding_length = len(eeg1_flattened) - len(upsampled_sleep_stages)
                    upsampled_sleep_stages = np.pad(upsampled_sleep_stages, (0, padding_length), mode='constant')

            extracted_data = {
                'sleepStage': upsampled_sleep_stages,
                'EEG1': eeg1_flattened,
                'EEG2': eeg2_flattened,
                'EMG': emg_flattened
            }

            df = pd.DataFrame(extracted_data)

            # Save DataFrame to a CSV file
            if not os.path.exists(output_directory_path):
                os.makedirs(output_directory_path)
            output_file_path = os.path.join(output_directory_path, base_filename + '.csv')
            df.to_csv(output_file_path, index=False)
            print(f'Saved CSV to: {output_file_path}')

if __name__ == "__main__":
    print("MAT to CSV Conversion Utility")
    print("-----------------------------")
    
    # These inputs are only requested when the script is run directly
    train_test_or_to_score = input("Enter dataset type ('train', 'test', or 'to_score'): ")
    base_directory = input(f"Enter the base somnotate directory path without quotes (e.g., Z:/somnotate): ")
    output_directory_path = os.path.join(base_directory, f"{train_test_or_to_score}_set", f"{train_test_or_to_score}_csv_files")
    sampling_rate = int(input("Enter the sampling rate in Hz (e.g., 512): "))
    sleep_stage_resolution = int(input("Enter the sleep stage resolution in seconds (e.g., 10): "))

    # Allow user to enter file paths one by one
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
    
    if file_paths:
        print(f"\nWill process {len(file_paths)} files and save to {output_directory_path}")
        proceed = input("Continue? (y/n): ").strip().lower() == 'y'
        if proceed:
            mat_to_csv(file_paths, output_directory_path, sampling_rate, sleep_stage_resolution)
        else:
            print("Operation cancelled.")
    else:
        print("No files to convert. Exiting.")
