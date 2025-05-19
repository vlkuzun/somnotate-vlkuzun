import pandas as pd
import numpy as np
import pyedflib
import os

def generate_edf_and_visbrain_formats(mouse_ids, sessions, recordings, extra_info, test_train_or_to_score, base_directory, sampling_rate):
    '''
    Generate EDF and Visbrain stage duration format files from CSV files, respectively for EEG and EMG data and sleep stage annotations.
    
    Inputs:
    mouse_ids: list of str, mouse IDs
    sessions: list of str, session IDs
    recordings: list of str, recording IDs
    extra_info: str, additional information to include in the output filenames for differentiation (optional)
    test_train_or_to_score: str, 'test', 'train' or 'to_score' to specify which dataset to process
    base_directory: str, path to the base directory where the CSV files are stored and where the output EDF and annotations files should be saved

    Outputs:
    EDF files and annotations in Visbrain stage duration format are saved in the 'edfs' and '{test_train_or_to_score}_manual_annotation' directories, respectively.
    '''

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
                        'sample_rate': sampling_rate,
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
                start_time = 10 / sampling_rate

                for i, label in enumerate(df["sleepStage"]):
                    current_time = i / sampling_rate
                    if label != current_stage:
                        if current_stage is not None:
                            annotations.append((start_time, current_time, current_stage))
                        current_stage = label
                        start_time = current_time
                annotations.append((start_time, len(df) / sampling_rate, current_stage))

                # Write annotations to a text file
                last_time_value = annotations[-1][1]
                with open(visbrain_file, "w") as f:
                    f.write(f"*Duration_sec    {last_time_value}\n")
                    f.write("*Datafile\tUnspecified\n")
                    for start, end, stage in annotations:
                        stage_label = {1: "awake", 2: "non-REM", 3: "REM", 4: 'ambiguous', 5: 'doubt'}.get(stage, "Undefined")
                        f.write(f"{stage_label}    {end}\n")

                print(f"EDF file and annotations created successfully for {mouse_id}, {session}, {recording} with extra info '{extra_info}'.")

if __name__ == "__main__":
    mouse_ids = input("Enter mouse IDs comma-separated without spaces (e.g., sub-001,sub-002): ").split(',')
    sessions = input("Enter session IDs comma-separated without spaces (e.g., ses-01,ses-02): ").split(',')
    recordings = input("Enter recording IDs comma-separated without spaces (e.g., recording-01): ").split(',')
    extra_info = input("Enter any extra details about the recording for differentiation (leave blank if not applicable): ").strip()
    test_train_or_to_score = input("Enter dataset type ('test', 'train', or 'to_score'): ").strip()
    base_directory = input("Enter the base somnotate directory path without quotes (e.g., Z:/somnotate): ").strip()
    sampling_rate = float(input("Enter the sampling rate in Hz (e.g., 512.0): "))

    generate_edf_and_visbrain_formats(mouse_ids, sessions, recordings, extra_info, test_train_or_to_score, base_directory, sampling_rate)
