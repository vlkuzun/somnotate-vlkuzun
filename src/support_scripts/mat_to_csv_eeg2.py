import numpy as np
import pandas as pd
import h5py
import os

# Ask for user input in the terminal for paths and recording parameters
train_test_or_to_score = input("Enter dataset type ('train', 'test', or 'to_score'): ")
output_directory_path = input(f"Enter the output directory path for {train_test_or_to_score} CSV files, without quotes (e.g., Z:/somnotate/to_score_set/to_score_csv_files): ")
sampling_rate = int(input("Enter the sampling rate in Hz (e.g., 512): "))
sleep_stage_resolution = int(input("Enter the sleep stage resolution in seconds (e.g., 10): "))

# Allow user to enter file paths one by one
file_paths = []
while True:
    file_path = input("Enter the full path of a .mat file to convert (or press Enter to finish): ")
    if file_path == "":
        break
    if os.path.isfile(file_path) and file_path.endswith('.mat'):
        file_paths.append(file_path)
    else:
        print("Invalid file path. Please enter a valid .mat file path.")

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
            eeg2_data, emg_data, sleep_stages = None, None, None

            # Iterate over all keys in the HDF5 file to extract data
            for key in raw_data.keys():
                if key.endswith('_EEG_EEG2A_B') or key.endswith('_EEGorig'):
                    eeg2_data = np.array(raw_data[key]['values'])
                elif key.endswith('_EMG_EMG'):
                    emg_data = np.array(raw_data[key]['values'])
                elif key.endswith('_Stage_1_'):
                    sleep_stages = np.array(raw_data[key]['codes'])
                    sleep_stages = sleep_stages[0, :]
            
            # Check if data was found
            if eeg2_data is not None:
                print("EEG2 data extracted successfully.")
            if emg_data is not None:
                print("EMG data extracted successfully.")
            if sleep_stages is not None:
                print("Sleep stage data extracted successfully.")

            # Format data for saving to a CSV file
            eeg2_flattened = eeg2_data.flatten()
            emg_flattened = emg_data.flatten()
            assert eeg2_flattened.shape == emg_flattened.shape, "The flattened shapes of the EEG and EMG data do not match"

            # Upsample sleep stages to match EEG/EMG data resolution
            upsampled_sleep_stages = np.repeat(sleep_stages, sampling_rate * sleep_stage_resolution)
            if len(upsampled_sleep_stages) != len(eeg2_flattened):
                print(f"Length of upsampled sleep stages ({len(upsampled_sleep_stages)}) does not match length of EEG data ({len(eeg2_flattened)})")
                if len(upsampled_sleep_stages) > len(eeg2_flattened):
                    upsampled_sleep_stages = upsampled_sleep_stages[:len(eeg2_flattened)]
                else:
                    padding_length = len(eeg2_flattened) - len(upsampled_sleep_stages)
                    upsampled_sleep_stages = np.pad(upsampled_sleep_stages, (0, padding_length), mode='constant')

            extracted_data = {
                'sleepStage': upsampled_sleep_stages,
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
    print("Starting main block...")
    mat_to_csv(file_paths, output_directory_path, sampling_rate, sleep_stage_resolution)
