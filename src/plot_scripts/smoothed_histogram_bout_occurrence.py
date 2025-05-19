
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # Gaussian smoothing
from functions_for_somno_QM_checks_volkan import get_bout_durations

def compare_histograms_smoothed(file_1, file_2, sampling_rate=1):
    """
    Compares histograms of bout durations between two CSV files with 5-second bins,
    compresses high-frequency ranges using logarithmic scaling,
    and plots smoothed lines connecting histogram points.
    Only considers bout durations up to 500 seconds.

    Input:
        file_1 (str): Path to the first CSV file
        file_2 (str): Path to the second CSV file
        sampling_rate (int): Sampling rate of the data in Hz (default is 1Hz)
        
    Output:
        Displays a histogram comparison with smoothed overlaid lines.
    """
    
    # Read the two CSV files into DataFrames
    df_1 = pd.read_csv(file_1)
    df_2 = pd.read_csv(file_2)
    
    # Calculate bout durations for both files using the get_bout_durations() function
    bout_durations_dict_1 = get_bout_durations(df_1, sampling_rate, "file_1")
    bout_durations_dict_2 = get_bout_durations(df_2, sampling_rate, "file_2")
    
    # Extract bout durations only
    bout_durations_1 = bout_durations_dict_1["file_1"]['BoutDuration'].values
    bout_durations_2 = bout_durations_dict_2["file_2"]['BoutDuration'].values

    # Restrict data to only include bout durations up to 500 seconds
    bout_durations_1 = bout_durations_1[bout_durations_1 <= 500]
    bout_durations_2 = bout_durations_2[bout_durations_2 <= 500]

    # Set up histogram parameters
    bins = np.arange(0, 505, 15)  # 15-second bins up to 500 seconds

    # Plotting histograms
    plt.figure(figsize=(14, 10))
    
    # Compute histogram counts
    counts_1, _ = np.histogram(bout_durations_1, bins=bins)
    counts_2, _ = np.histogram(bout_durations_2, bins=bins)

    # Smooth the histogram counts using Gaussian filter
    smooth_counts_1 = gaussian_filter1d(counts_1, sigma=3)  # Gaussian smoothing with sigma=3
    smooth_counts_2 = gaussian_filter1d(counts_2, sigma=3)

    # Overlay smoothed lines
    x_bins_center = (bins[:-1] + bins[1:]) / 2  # Calculate the center of each bin for plotting
    plt.plot(x_bins_center, smooth_counts_1, label="Somnotate", color='blue', lw=2)
    plt.plot(x_bins_center, smooth_counts_2, label="Manual", color='orange', lw=2)

    # Set logarithmic scaling for the y-axis
    #plt.yscale('log')

    # Customize the plot
    plt.xlabel('Bout Duration (seconds)', fontsize=18)
    plt.ylabel('Occurrence', fontsize=18)
    plt.title('Comparison of Bout Duration Occurrence Between Somnotate and Manual Scoring Smoothed', fontsize=20, pad=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(axis='y', alpha=0.5)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Finalize the plot
    plt.tight_layout()
    plt.show()

compare_histograms_smoothed(
    file_1="/Volumes/harris/volkan/sleep_profile/downsample_auto_score/sub-010_ses-01_recording-01_time-0-69h_sr-1hz.csv",
    file_2="/Volumes/harris/volkan/sleep_profile/downsample_manual_score/sub-010_ses-01_recording-01_time-0-69h_manual_sr-1hz.csv",
    sampling_rate=1  # or other sampling rates if required
)


