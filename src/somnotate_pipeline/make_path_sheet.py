import glob
import os
import pandas as pd


def make_train_and_test_sheet(train_test_or_to_score, base_directory, sampling_rate):
    """Create a CSV file containing the file paths of the raw signals, manual annotations, and preprocessed signals.
    Input:
    - train_test_or_to_score: str, the dataset type ('train', 'test', or 'to_score').
    - base_directory: str, the base directory path containing the dataset.
    - sampling_rate: float, the sampling rate of the signals.
    Output:
    - A CSV file containing the file paths of the raw signals, manual annotations, and preprocessed signals.
    """

    csv_file_directory = os.path.join(base_directory, f"{train_test_or_to_score}_set")

    # define paths to the raw signals
    raw_signals_path = os.path.join(base_directory, f"{train_test_or_to_score}_set/edfs")
    edf_files = glob.glob(os.path.join(raw_signals_path, '*.edf'))

    # define paths to the manual annotations
    manual_annotations_path = os.path.join(base_directory, f"{train_test_or_to_score}_set/{train_test_or_to_score}_manual_annotation")
    vis_files = glob.glob(os.path.join(manual_annotations_path, '*.txt'))
    assert len(vis_files) == len(edf_files), "The number of manual annotations does not match the number of raw signals."
    print(f'Including {vis_files} in the {train_test_or_to_score} set.')

    csv_data_list = []
    print(f'Including {edf_files} in the {train_test_or_to_score} set.')

    # Iterate over each edf file
    for edf_file in edf_files:
        base_filename = os.path.splitext(os.path.basename(edf_file))[0]
        print(base_filename)

        # define paths to the preprocessed signals
        preprocessed_signals_path = os.path.join(base_directory, f"{train_test_or_to_score}_set/preprocessed_signals")
        if not os.path.exists(preprocessed_signals_path):
            os.makedirs(preprocessed_signals_path)
        file_path_preprocessed_signals = os.path.join(preprocessed_signals_path, base_filename + '.npy')

        # define paths to the manual annotations corresponding to the current edf file
        recording_identifier= '_'.join(base_filename.split('_')[1:]) # remove the first string from the base filename
        corresponding_manual_annotation_file = None
        for vis_file in vis_files:
            if recording_identifier in vis_file:  # Match based on the base filename
                corresponding_manual_annotation_file = os.path.normpath(vis_file)
                break

        if corresponding_manual_annotation_file is None:
            raise ValueError(f"No matching manual annotation found for {base_filename}")


        # define paths to the automated annotations
        automated_annotations_path = os.path.join(base_directory, f"{train_test_or_to_score}_set/automated_annotation")
        if not os.path.exists(automated_annotations_path):
            os.makedirs(automated_annotations_path)
        file_path_automated_state_annotation = os.path.join(automated_annotations_path, 'automated_state_annotation' + base_filename + '.txt')
        file_path_automated_state_annotation = os.path.normpath(file_path_automated_state_annotation)

        # define paths to the refines annotations
        refined_annotations_path = os.path.join(base_directory, f"{train_test_or_to_score}_set/refined_annotations")
        if not os.path.exists(refined_annotations_path):
            os.makedirs(refined_annotations_path)
        file_path_refined_state_annotation = os.path.join(refined_annotations_path, 'refined_state_annotation' + base_filename + '.txt')
        file_path_refined_state_annotation = os.path.normpath(file_path_refined_state_annotation)

        # define paths to review intervals 
        review_intervals_path = os.path.join(base_directory, f"{train_test_or_to_score}_set/int")
        if not os.path.exists(review_intervals_path):
            os.makedirs(review_intervals_path)
        file_path_review_intervals = os.path.join(review_intervals_path, 'review_intervals' + base_filename + '.txt')
        file_path_review_intervals = os.path.normpath(file_path_review_intervals)

        # define paths to manual artefact annotations
        manual_artefact_annotations_path = os.path.join(base_directory, f"{train_test_or_to_score}_set/manual_artefacts")
        if not os.path.exists(manual_artefact_annotations_path):
            os.makedirs(manual_artefact_annotations_path)
        file_path_manual_artefact_annotation = os.path.join(manual_artefact_annotations_path, 'man_art' + base_filename + '.txt')
        file_path_manual_artefact_annotation = os.path.normpath(file_path_manual_artefact_annotation)

        # define paths to automated artefact annotations
        automated_artefact_annotations_path = os.path.join(base_directory, f"{train_test_or_to_score}_set/auto_artefacts")
        if not os.path.exists(automated_artefact_annotations_path):
            os.makedirs(automated_artefact_annotations_path)
        file_path_automated_artefact_annotation = os.path.join(automated_artefact_annotations_path, 'auto_art' + base_filename + '.txt')
        file_path_automated_artefact_annotation = os.path.normpath(file_path_automated_artefact_annotation)

        # define paths to artefact intervals
        file_path_artefact_intervals = os.path.join(automated_artefact_annotations_path, 'art_in' + base_filename + '.txt')
        file_path_artefact_intervals = os.path.normpath(file_path_artefact_intervals)

        # define paths to missing values intervals 
        file_path_missing_values_intervals = os.path.join(automated_artefact_annotations_path, 'miss_int' + base_filename + '.txt')
        file_path_missing_values_intervals = os.path.normpath(file_path_missing_values_intervals)

        # define paths to state probabilites
        state_probabilities_path = os.path.join(base_directory, f"{train_test_or_to_score}_set/state_probabilities")
        if not os.path.exists(state_probabilities_path):
            os.makedirs(state_probabilities_path)
        file_path_state_probabilities = os.path.join(state_probabilities_path, 'state_probabilities' + base_filename + '.npz')

        csv_data = {
            'file_path_raw_signals' : edf_file,
            'eeg1_signal_label': 'EEG1',
            'eeg2_signal_label': 'EEG2',
            'emg_signal_label': 'EMG',
            'sampling_frequency_in_hz': sampling_rate,
            'file_path_preprocessed_signals': file_path_preprocessed_signals,
            'file_path_manual_state_annotation': corresponding_manual_annotation_file,
            'file_path_automated_state_annotation': file_path_automated_state_annotation,
            'file_path_refined_state_annotation': file_path_refined_state_annotation,
            'file_path_review_intervals': file_path_review_intervals,
            'file_path_manual_artefact_annotation': file_path_manual_artefact_annotation,
            'file_path_automated_artefact_annotation': file_path_automated_artefact_annotation,
            'file_path_artefact_intervals': file_path_artefact_intervals,
            'file_path_missing_values_intervals': file_path_missing_values_intervals,
            'file_path_state_probabilities': file_path_state_probabilities
        }

        
        # Append the current file's data to the list
        csv_data_list.append(csv_data)

    # Save the CSV data to a file
    df = pd.DataFrame(csv_data_list)
    output_path = os.path.join(csv_file_directory, f"{train_test_or_to_score}_sheet" + ".csv")
    df.to_csv(output_path, index=False)
    return output_path


# Only run this block if the script is executed directly, not when imported
if __name__ == '__main__':
    # Ask for user input directly only when run as a script
    train_test_or_to_score = input("Enter dataset type ('train', 'test', or 'to_score'): ")
    base_directory = input("Enter the somnotate base directory path without quotes (e.g., Z:/somnotate): ")
    sampling_rate = float(input("Enter the sampling rate (e.g., 512.0): "))
    
    make_train_and_test_sheet(train_test_or_to_score, base_directory, sampling_rate)



