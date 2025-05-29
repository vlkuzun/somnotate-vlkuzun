import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set global style for publication
plt.rcParams.update({
    'font.family': 'Arial',         # Use Arial (or Helvetica as fallback)
    'font.size': 10,                # General font size
    'axes.labelsize': 12,           # Axis label size
    'axes.titlesize': 12,           # Title size
    'xtick.labelsize': 10,          # X tick label size
    'ytick.labelsize': 10,          # Y tick label size
    'legend.fontsize': 10,          # Legend text size
    'figure.dpi': 300,              # High-res output
    'savefig.dpi': 600,             # High-res when saving
    'figure.figsize': [4, 3],       # Width x Height in inches
    'axes.linewidth': 1,            # Thinner axis borders
    'pdf.fonttype': 42,             # Embed fonts properly in PDFs
    'ps.fonttype': 42
})

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
    1: '#E69F00',  # Golden yellow for WAKE
    2: '#56B4E9',  # Sky blue for non-REM
    3: '#CC79A7',  # Pink/magenta for REM
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

    ax.set_title(f'{label}')
    ax.set_yticks(np.arange(-500,505,500))
    ax.set_ylabel('EEG (μV)')
    
    # Only move y-axis to origin, keep x-axis at bottom
    ax.spines['left'].set_position(('data', 0))
    # Remove the positioning of the bottom spine to keep it at the bottom
    
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Create subplots - adjusted figure size for publication
fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True, sharey=True)

# Plot EEG with the corresponding sleep stages for each classifier using the downsampled data
plot_eeg(axes[0], df_slice_downsampled['EEG2'], df_slice_downsampled['sleepStage_somnotate'], 'somnotate')
plot_eeg(axes[1], df_slice_downsampled['EEG2'], df_slice_downsampled['sleepStage_fp'], 'fp')
plot_eeg(axes[2], df_slice_downsampled['EEG2'], df_slice_downsampled['sleepStage_bh'], 'bh')
plot_eeg(axes[3], df_slice_downsampled['EEG2'], df_slice_downsampled['sleepStage_vu'], 'vu')

# Calculate the time range from the data
time_range = len(df_slice_downsampled) / (sampling_rate / downsample_factor)
min_time = 0  # Starting time
max_time = time_range  # Ending time

# Add a unified x-axis label
fig.text(0.5, 0.02, 'Time (seconds)', ha='center')

# Create a unified legend for sleep stages
handles = [plt.Line2D([0], [0], color=colors[1], lw=2, label='Wake'),
           plt.Line2D([0], [0], color=colors[2], lw=2, label='NREM'),
           plt.Line2D([0], [0], color=colors[3], lw=2, label='REM')]
labels = [f'Stage {stage}' for stage in colors]

# Adjust legend position to be above the plots
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.91, 1.0), ncol=1)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to leave space for the legend and x-axis label

# For all subplots, adjust the axis positions
for i, ax in enumerate(axes):
    # Hide x-axis labels for all but the bottom subplot
    if i < len(axes) - 1:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    else:
        ax.tick_params(axis='x', which='both', labelbottom=True)
    
    # Set the x-axis limits to match the actual data extent
    ax.set_xlim(min_time, max_time)
    
    # Adjust position to ensure axis starts at origin
    ax.set_position([0.125, ax.get_position().y0, 0.775, ax.get_position().height])

# Save figure - DPI is now controlled by the global parameters
plt.savefig('/Volumes/harris/volkan/somnotate-vlkuzun/plots/performance_vs_manual/somno_vs_manual_eeg_snippet.png', bbox_inches='tight')
plt.savefig('/Volumes/harris/volkan/somnotate-vlkuzun/plots/performance_vs_manual/somno_vs_manual_eeg_snippet.pdf', format='pdf', bbox_inches='tight')

plt.show()