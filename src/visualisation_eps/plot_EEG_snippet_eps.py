import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set global font size
plt.rcParams.update({'font.size': 20})  # Increase the global font size

# Load the data
path_combined_df = '/Volumes/harris/Francesca/somnotate/checking_accuracy/combined_data.csv'
combined_df = pd.read_csv(path_combined_df)

sampling_rate = 512

# Extract a slice between second 2900 and 8000
start_time = 2900 * sampling_rate  # Convert seconds to samples
end_time = 8000 * sampling_rate    # Convert seconds to samples

# Get the slice of the dataframe
df_slice = combined_df.iloc[start_time:end_time]

# Downsample the data for better visualization
downsample_factor = 1  # Adjust this factor based on how much you want to downsample
df_slice_downsampled = df_slice.iloc[::downsample_factor].reset_index(drop=True)

# Define colors for each sleep stage using tones of blue
colors = {
    1: '#B3CDE3',  # Lightest blue for WAKE
    2: '#6497B1',  # Medium blue for non-REM
    3: '#005B96',  # Dark blue for REM
}

# Plot function for the EEG signal
def plot_eeg(ax, eeg_data, sleep_stages, label):
    time = np.arange(len(eeg_data)) / (sampling_rate / downsample_factor)  # Time in seconds
    
    current_stage = None
    start_index = 0

    # Loop through each index in sleep stages
    for i in range(len(sleep_stages)):
        stage = sleep_stages[i]
        
        # If we encounter a change in sleep stage
        if stage != current_stage:
            if current_stage is not None:
                # If we are not at the first stage, plot the previous segment
                ax.plot(time[start_index:i], eeg_data[start_index:i], color=colors.get(current_stage, 'gray'), lw=0.5)
                
            # Update to the new stage
            current_stage = stage
            start_index = i

    # Plot the last segment
    if current_stage is not None:
        ax.plot(time[start_index:], eeg_data[start_index:], color=colors.get(current_stage, 'gray'), lw=0.5)

    ax.set_title(f'{label}', fontsize=24)  # Increased font size
    ax.set_yticks(np.arange(-500,505,500))
    ax.set_ylabel('EEG (μV)', fontsize=22)  # Increased font size
    ax.tick_params(axis='y', labelsize=20)  # Increased font size
    ax.tick_params(axis='x', labelsize=20)  # Increased font size

# Create subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)  # Slightly larger figure

# Plot EEG with the corresponding sleep stages for each classifier using the downsampled data
plot_eeg(axes[0], df_slice_downsampled['EEG2'], df_slice_downsampled['sleepStage_somnotate'], 'somnotate')
plot_eeg(axes[1], df_slice_downsampled['EEG2'], df_slice_downsampled['sleepStage_fp'], 'fp')
plot_eeg(axes[2], df_slice_downsampled['EEG2'], df_slice_downsampled['sleepStage_bh'], 'bh')
plot_eeg(axes[3], df_slice_downsampled['EEG2'], df_slice_downsampled['sleepStage_vu'], 'vu')

# Add a unified x-axis label
fig.text(0.56, 0.02, 'Time (seconds)', ha='center', fontsize=24)  # Increased font size

# Create a unified legend for sleep stages
handles = [plt.Line2D([0], [0], color=colors[1], lw=2, label='Wake'),
           plt.Line2D([0], [0], color=colors[2], lw=2, label='NREM'),
           plt.Line2D([0], [0], color=colors[3], lw=2, label='REM')]
labels = [f'Stage {stage}' for stage in colors]

# Adjust legend position to be above the plots
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.93, 1.01), ncol=1, fontsize=18)  # Increased font size

plt.tight_layout(rect=[0, 0.05, 1, 0.93])  # Adjust layout to leave space for the legend and x-axis label

# Save figure as EPS (vector format)
plt.savefig('/Volumes/harris/volkan/somnotate-vlkuzun/plots/performance_vs_manual/eps/somno_vs_manual_eeg_snippet.eps', format='eps', bbox_inches='tight')

plt.show()