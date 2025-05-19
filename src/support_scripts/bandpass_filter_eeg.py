import pandas as pd
import numpy as np
from scipy import signal
import os

def bandpass_filter_eeg(input_file, output_dir, lowcut, highcut, fs=512, order=2):
    """
    Filter EEG channels and save to new pickle file.
    If lowcut is 0 or None, applies lowpass filter instead of bandpass.
    
    Args:
        input_file (str): Path to input pickle file
        output_dir (str): Directory to save filtered data
        lowcut (float): Lower frequency bound (0 or None for lowpass)
        highcut (float): Upper frequency bound
        fs (int): Sampling frequency in Hz
        order (int): Filter order
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    eeg_data = pd.read_pickle(input_file)
    
    # Design filter
    nyquist = fs / 2
    if lowcut in [0, None]:
        high = highcut / nyquist
        b, a = signal.butter(order, high, btype='low')
    else:
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
    
    # Filter EEG channels
    for channel in ['EEG1', 'EEG2']:
        if channel in eeg_data.columns:
            filtered_data = signal.filtfilt(b, a, eeg_data[channel])
            eeg_data[channel] = filtered_data
    
    # Generate output filename
    basename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{basename}_filtered_{lowcut}-{highcut}Hz.pkl")
    
    # Save filtered data
    eeg_data.to_pickle(output_file)
    
    return output_file

# Example usage:
if __name__ == "__main__":
    input_files = [
        #'/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-007_ses-01_recording-01.pkl',
        '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-010_ses-01_recording-01.pkl',
        #'/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-011_ses-01_recording-01.pkl',
        #'/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-015_ses-01_recording-01.pkl',
        #'/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-016_ses-02_recording-01.pkl',
        #'/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal/eeg_data_sub-017_ses-01_recording-01.pkl'
    ]
    output_dir = '/Volumes/harris/somnotate/to_score_set/pickle_eeg_signal'
    
    # Filter 0-50 Hz bandpass as an example
    for file in input_files:
        filtered_file = bandpass_filter_eeg(file, output_dir, 0.5, 50)
        print(f"Saved filtered data to: {filtered_file}")