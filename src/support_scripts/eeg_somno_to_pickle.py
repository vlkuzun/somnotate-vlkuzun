import pandas as pd

def load_and_save_data():
    while True:
        # Prompt the user for subject ID
        subject_id = input("Enter the subject ID (e.g., sub-007) or 'exit' to stop: ").strip()
        if subject_id.lower() == 'exit':
            print("Exiting the process.")
            break

        # Prompt the user for file locations
        eeg_file = input(f"Enter the location of the EEG data CSV file for {subject_id}: ").strip()
        somno_file = input(f"Enter the location of the Somno scoring CSV file for {subject_id}: ").strip()

        # Load the files
        try:
            eeg_data = pd.read_csv(eeg_file)
            somno_scoring = pd.read_csv(somno_file)
        except Exception as e:
            print(f"Error loading files: {e}")
            continue

        # Ask user for output locations
        eeg_pickle_path = input(f"Enter the location to save EEG data pickle for {subject_id} (e.g., path/to/eeg_data_{subject_id}.pkl): ").strip()
        somno_pickle_path = input(f"Enter the location to save Somno scoring pickle for {subject_id} (e.g., path/to/somno_scoring_{subject_id}.pkl): ").strip()

        # Save the files as pickle
        try:
            eeg_data.to_pickle(eeg_pickle_path)
            print(f"EEG data for {subject_id} saved as pickle file at {eeg_pickle_path}.")

            somno_scoring.to_pickle(somno_pickle_path)
            print(f"Somno scoring for {subject_id} saved as pickle file at {somno_pickle_path}.")
        except Exception as e:
            print(f"Error saving pickle files: {e}")

if __name__ == "__main__":
    load_and_save_data()
