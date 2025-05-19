import pandas as pd

def update_sleep_stage(csv_file, output_csv, sampling_rate=512, duration_sec=10):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Calculate the number of samples to change based on the sampling rate and duration
    num_samples = sampling_rate * duration_sec

    # Ensure there are enough samples in the DataFrame
    if len(df) < num_samples:
        raise ValueError("CSV file has fewer rows than needed for the specified duration.")
    
    # Update the sleepStage values in the first `num_samples` rows
    df.loc[:num_samples-1, 'sleepStage'] = 1
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Updated file saved to {output_csv}")

# Example usage
csv_file = "/ceph/harris/somnotate/to_score_set/vis_back_to_csv/annotations_visbrain_sub-010_ses-01_recording-01_time-0-69h.csv"
output_csv = "/ceph/harris/somnotate/to_score_set/vis_back_to_csv/annotations_visbrain_sub-010_ses-01_recording-01_time-0-69h_update.csv"
update_sleep_stage(csv_file, output_csv)