import os
import pandas as pd
import datetime

def get_stage_numeric(stage_label):
    '''
    Converts the sleep stage label to a numeric value.
    '''
    return {"awake": 1, "non-REM": 2, "REM": 3, "ambiguous": 4, "doubt": 5, "Undefined": 0}.get(stage_label, 0)

def read_visbrain_file(visbrain_file, sampling_rate, start_time):
    '''
    Reads the Visbrain TXT file and returns a list of sleep stages per sample.
    '''
    annotations = []
    
    with open(visbrain_file, "r") as f:
        lines = f.readlines()
    
    # Ignore the header lines
    assert lines[0].startswith("*") and lines[1].startswith("*"), "Invalid Visbrain file format"
    recording_duration = lines[0].strip().split()[1]  # Extract the recording duration from the first line
    lines = lines[2:]
    print(f"Recording duration: {recording_duration} seconds")

    sleep_stages = []
    timestamps = []
    
    start_time_sec = 0  # Tracks the previous end time in seconds

    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        
        stage_label, end_time = parts
        stage_numeric = get_stage_numeric(stage_label)
        end_time = float(end_time)
        
        # Calculate number of samples for this stage
        duration_in_samples = round((end_time - start_time_sec) * sampling_rate)
        
        # Generate timestamps and sleep stages
        for sample_index in range(duration_in_samples):
            sleep_stages.append(stage_numeric)
            timestamps.append(start_time + datetime.timedelta(seconds=len(sleep_stages) / sampling_rate))
        
        start_time_sec = end_time  # Update start time for the next stage
    
    return sleep_stages, recording_duration, timestamps

def visbrain_to_csv(visbrain_file, output_csv, sampling_rate, start_time):
    '''
    Read the sleep stages from a Visbrain TXT file and save them to a CSV file.
    '''
    sleep_stages, recording_duration, timestamps = read_visbrain_file(visbrain_file, sampling_rate, start_time)
    
    # Save the reconstructed data into a DataFrame
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "sleepStage": sleep_stages
    })
    print('Dataframe length (samples):', len(df), 'Dataframe duration (secs):', len(df)/sampling_rate, 'Recording duration (secs):', recording_duration)
    
    # Write to a CSV file
    df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    df.to_csv(output_csv, index=False)

    print(f"CSV file saved successfully: {output_csv}")

def compare_csv_and_visbrain(csv_file, visbrain_file, sampling_rate, start_time):
    '''
    Compare the sleep stages from a CSV file and a Visbrain TXT file.
    '''
    df = pd.read_csv(csv_file)
    csv_stages = df["sleepStage"].tolist()
    
    # Load the stages from the Visbrain TXT file
    visbrain_stages, _, _ = read_visbrain_file(visbrain_file, sampling_rate, start_time)

    # Ensure both have the same length
    if len(csv_stages) != len(visbrain_stages):
        print(f"Length mismatch: CSV has {len(csv_stages)} samples, Visbrain file has {len(visbrain_stages)} samples.")
        return
    
    # Check for mismatches between CSV and Visbrain stages
    mismatches = []
    for i, (csv_stage, visbrain_stage) in enumerate(zip(csv_stages, visbrain_stages)):
        if csv_stage != visbrain_stage:
            mismatches.append((i, csv_stage, visbrain_stage))
    
    # Report mismatches, i.e. indexes where the stages differ
    if mismatches:
        print(f"Found {len(mismatches)} mismatches between the CSV and Visbrain file:")
        for idx, csv_stage, visbrain_stage in mismatches[:10]:  # Print first 10 mismatches
            print(f"Mismatch at index {idx}: CSV={csv_stage}, Visbrain={visbrain_stage}")
        print(f"...and {len(mismatches)} total mismatches.")
    else:
        print("The CSV and Visbrain files match perfectly.")

def main():
    # Get the sampling rate
    sampling_rate = int(input("Enter the desired sampling rate (in Hz): "))

    # Get the start time in the format "YYYY-MM-DD HH:MM:SS"
    while True:
        start_time_str = input("Enter the start time in the format YYYY-MM-DD HH:MM:SS: ")
        try:
            start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            break
        except ValueError:
            print("Invalid format. Please enter the start time in the format YYYY-MM-DD HH:MM:SS.")
    
    # Ask user for the list of Visbrain files
    visbrain_paths = []
    while True:
        path = input("Enter the path to a Visbrain TXT file (or press Enter to finish): ")
        if not path:
            break
        visbrain_paths.append(path)

    # Ask user for the output directory
    output_directory = input("Enter the directory path for the output CSV files: ")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each Visbrain file and save the CSV in the output directory
    for visbrain_file in visbrain_paths:
        # Derive the output CSV filename from the Visbrain filename and sampling rate
        output_filename = f"{os.path.splitext(os.path.basename(visbrain_file))[0]}_{sampling_rate}Hz.csv"
        output_csv = os.path.join(output_directory, output_filename)

        print(f"\nProcessing {visbrain_file} and saving to {output_csv}")
        visbrain_to_csv(visbrain_file, output_csv, sampling_rate, start_time)
        compare_csv_and_visbrain(output_csv, visbrain_file, sampling_rate, start_time)

if __name__ == "__main__":
    main()

print('Done')
