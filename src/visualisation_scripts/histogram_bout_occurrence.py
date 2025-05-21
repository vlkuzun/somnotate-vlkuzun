import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_for_somno_QM_checks import get_bout_durations

def compare_histograms(file_1, file_2, sampling_rate=1, output_path=None):
    """
    Compares histograms of bout durations between two CSV files with 60-second bins,
    and plots histogram bars in different colors for each file.
    Only considers bout durations up to 600 seconds.

    Input:
        file_1 (str): Path to the first CSV file
        file_2 (str): Path to the second CSV file
        sampling_rate (int): Sampling rate of the data in Hz (default is 1Hz)
        output_path (str): Path to save the figure (if None, figure will be displayed)
        
    Output:
        Displays or saves a histogram comparison with distinct bar colors.
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

    # Restrict data to only include bout durations up to 600 seconds
    bout_durations_1 = bout_durations_1[bout_durations_1 <= 600]
    bout_durations_2 = bout_durations_2[bout_durations_2 <= 600]

    # Set up histogram parameters
    bins = np.arange(0, 606, 60)  # 60-second bins up to 600 seconds

    # Plotting histograms
    plt.figure(figsize=(16, 14))

    # Plot histogram for file 1
    plt.hist(bout_durations_1, bins=bins, alpha=1, label="Somnotate", color='#1f77b4', edgecolor='black')

    # Plot histogram for file 2
    plt.hist(bout_durations_2, bins=bins, alpha=1, label="Manual", color='#ff7f0e', edgecolor='black')

    # Set logarithmic scaling for the y-axis
    #plt.yscale('log')

    # Customize the plot
    plt.xlabel('Bout Epoch (seconds)', fontsize=50, labelpad=10)
    plt.ylabel('Occurrence', fontsize=50, labelpad=20)
    plt.title('60 second bouts', fontsize=44, pad=30)
    #plt.xticks(np.arange(0, 125, 15), fontsize=44) # Define the tick location and range
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=44)
    plt.legend(fontsize=44)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Finalize the plot
    #plt.tight_layout()
    
    # Save the figure with high DPI or show it
    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"Figure saved to {output_path} with DPI=600")
    else:
        plt.show()

# Define the output path for saving the figure
output_file = "/Volumes/harris/volkan/somnotate/plots/bout_duration/occurrence_bout_duration/francesca_histogram_occurrence_bout_duration_somno_vs_manual_sub-010_60secbouts_600secs.png"

compare_histograms(
    file_1="/Volumes/harris/volkan/sleep_profile/francesca_sub-010/automated_state_annotationoutput_sub-010_ses-01_recording-01_sr-1hz.csv",
    file_2="/Volumes/harris/volkan/sleep_profile/francesca_sub-010/sub-010_ses-01_recording-01_data-sleepscore_fp_sr-1hz.csv",
    sampling_rate=1,  # or other sampling rates if required
    output_path=output_file  # Pass the output path to save the figure
)