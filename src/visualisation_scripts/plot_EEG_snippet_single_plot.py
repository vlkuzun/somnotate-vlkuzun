import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
path_combined_df = '/Volumes/harris/Francesca/somnotate/checking_accuracy/combined_data.csv'
combined_df = pd.read_csv(path_combined_df)

sampling_rate = 512  # Sampling rate in Hz

# Extract a slice between second 5720 and 5760
start_time = 4975  # Start time in seconds
end_time = 5000    # End time in seconds

# Convert seconds to sample indices using sampling rate
start_index = int(start_time * sampling_rate)
end_index = int(end_time * sampling_rate)

# Get the slice of the dataframe
df_slice = combined_df.iloc[start_index:end_index]

# Downsample the data for better visualization
downsample_factor = 5  # Change this value to downsample more heavily if necessary
df_slice_downsampled = df_slice.iloc[::downsample_factor].reset_index(drop=True)

# Define colors for each sleep stage using tones of blue
colors = {
    1: 'red',  # red for Awake
    2: 'blue',  # blue for non-REM
    #3: '#005B96',  # Dark blue for REM
}


def plot_single_eeg(ax, eeg_data, sleep_stages, label, time_window_start, shade_time_ranges):
    """
    Function to plot EEG signal with the corresponding sleep stages
    for a single EEG signal and its corresponding sleep stage classification.
    Allows shading of specified time ranges and legend creation.

    Parameters:
        ax (matplotlib axis): Axis to plot on.
        eeg_data (array-like): EEG signal data.
        sleep_stages (array-like): Corresponding sleep stage data.
        label (str): Label for the plot.
        time_window_start (float): The starting time in seconds.
        shade_time_ranges (list of tuples): List of time ranges to shade e.g., [(5735, 5745), (5747, 5755)]
    """
    # Calculate time in seconds for x-axis based on indices and sampling rate
    time = np.arange(len(eeg_data)) / (sampling_rate / downsample_factor) + time_window_start

    current_stage = None
    start_index = 0

    # Loop through indices to segment by changes in the sleep stages
    for i in range(len(sleep_stages)):
        stage = sleep_stages[i]

        # If we encounter a change in stage, plot the previous section
        if stage != current_stage:
            if current_stage is not None:
                ax.plot(
                    time[start_index:i],
                    eeg_data[start_index:i],
                    color=colors.get(current_stage, 'gray'),
                    lw=0.5,
                )

            # Update stage tracking
            current_stage = stage
            start_index = i

    # Plot the final segment
    if current_stage is not None:
        ax.plot(
            time[start_index:],
            eeg_data[start_index:],
            color=colors.get(current_stage, 'gray'),
            lw=0.5,
        )

    # Shade the specified time ranges
    for shade_range in shade_time_ranges:
        ax.axvspan(shade_range[0], shade_range[1], color='gray', alpha=0.3)

    # Set the title and axis labels
    #ax.set_title(f'{label}', fontsize=22, pad=30)
    ax.set_ylabel('EEG (μV)', fontsize=22)  # Increased from 18
    ax.tick_params(axis='x', labelsize=20)  # Increased from 18
    ax.tick_params(axis='y', labelsize=20)  # Increased from 18

    # Create legend with "Wake" instead of "Awake"
    handles = [
        plt.Line2D([0], [0], color=colors[1], lw=2, label='Wake'),  # Changed from "Awake"
        plt.Line2D([0], [0], color=colors[2], lw=2, label='NREM'),
        #plt.Line2D([0], [0], color=colors[3], lw=2, label='REM'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=18)  # Increased from 14


# Create subplots for visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 4))  # Only plot 1 dataset

# Define time ranges to shade on the plot
shade_time_ranges = [(4988, 4991), (4994, 4996)]  # Two ranges to shade

# Choose EEG and corresponding sleep stage column to visualize
plot_single_eeg(
    ax,
    df_slice_downsampled['EEG2'].values,  # EEG signal data
    df_slice_downsampled['sleepStage_somnotate'].values,  # Corresponding sleep stage data
    label='Somnotate Annotation with Shaded Likelihood Scoring',
    time_window_start=start_time,  # Adjust x-axis start time
    shade_time_ranges=shade_time_ranges,  # Pass the ranges to shade
)

# Customize the plot
ax.set_xlabel('Time (seconds)', fontsize=22)

# Display the plot
plt.tight_layout()
plt.savefig('/Volumes/harris/volkan/somnotate/plots/likelihood_scoring/awake_nrem_likelihood_eeg_snippet_shading_sub-010.png', dpi=600, bbox_inches='tight')
plt.show()
