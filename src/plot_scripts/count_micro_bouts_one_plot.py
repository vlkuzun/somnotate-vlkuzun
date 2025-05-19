import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
def load_data(file_path):
    """Load CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

# Count micro bouts
def count_micro_bouts(df):
    """Count micro bouts of wake, non-REM, and REM sleep throughout the recording."""
    micro_bouts = {1: 0, 2: 0, 3: 0}  # Initialize counters for wake, non-REM, and REM
    bout_start = None
    current_stage = None

    for index, row in df.iterrows():
        if row['sleepStage'] in [1, 2, 3]:
            if row['sleepStage'] != current_stage:
                if bout_start is not None and index - bout_start < 15:
                    micro_bouts[current_stage] += 1
                bout_start = index
                current_stage = row['sleepStage']
        elif bout_start is not None:
            if index - bout_start < 15:
                micro_bouts[current_stage] += 1
            bout_start = None

    # Handle last bout
    if bout_start is not None and len(df) - bout_start < 15:
        micro_bouts[current_stage] += 1

    return micro_bouts

def create_grouped_bar_chart(file_paths, output_file):
    """Create a grouped bar chart for micro bouts by sleep stage."""
    labels = ['Somnotate', 'Manual']
    stages = ['Wake', 'Non-REM', 'REM']
    colors = ['#1f77b4', '#ff7f0e']  # Blue and Orange

    # Initialize lists to store bout counts
    somnotate_bouts = {1: 0, 2: 0, 3: 0}
    averaged_bouts = {1: 0, 2: 0, 3: 0}

    # Process Somnotate file
    somnotate_file = file_paths[0]
    somnotate_bouts = count_micro_bouts(load_data(somnotate_file))

    # Process other files and compute averages
    other_files = file_paths[1:]
    temp_bouts = {1: [], 2: [], 3: []}

    for file_path in other_files:
        bouts = count_micro_bouts(load_data(file_path))
        for stage in [1, 2, 3]:
            temp_bouts[stage].append(bouts[stage])

    for stage in [1, 2, 3]:
        averaged_bouts[stage] = sum(temp_bouts[stage]) / len(temp_bouts[stage])

    # Prepare data for plotting
    data = {
        'Wake': [somnotate_bouts[1], averaged_bouts[1]],
        'Non-REM': [somnotate_bouts[2], averaged_bouts[2]],
        'REM': [somnotate_bouts[3], averaged_bouts[3]]
    }

    x = range(len(stages))  # Positions for the groups

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Significantly increase font size for all text elements
    TITLE_SIZE = 32
    LABEL_SIZE = 28  
    TICK_SIZE = 26
    LEGEND_SIZE = 28

    bar_width = 0.35
    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [data[stage][i] for stage in stages]
        ax.bar([pos + i * bar_width for pos in x], values, bar_width, label=label, color=color)

    # Formatting the plot
    ax.set_xticks([pos + bar_width / 2 for pos in x])
    ax.set_xticklabels(['Wake', 'NREM','REM'], fontsize=LABEL_SIZE)
    ax.set_ylabel('Occurrence', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=1, length=4)

    # Fix for y-tick labels
    yticks = np.arange(0, 29, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=TICK_SIZE)
    
    ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make remaining spines standard thickness (not thicker)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

# Main function
def main():
    file_paths = [
        '/Volumes/harris/volkan/sleep_profile/francesca_sub-010/automated_state_annotationoutput_sub-010_ses-01_recording-01_sr-1hz.csv',
        '/Volumes/harris/volkan/sleep_profile/francesca_sub-010/sub-010_ses-01_recording-01_data-sleepscore_fp_sr-1hz.csv',
        '/Volumes/harris/volkan/sleep_profile/francesca_sub-010/sub-010_ses-01_recording-01_data-sleepscore_vu_sr-1hz.csv',
        '/Volumes/harris/volkan/sleep_profile/francesca_sub-010/sub-010_ses-01_recording-01_export(HBH)_sr-1hz.csv'
    ]
    output_file = '/Volumes/harris/volkan/somnotate/plots/bout_duration/microbouts/micro_bout_bar_charts_one_plot.png'
    create_grouped_bar_chart(file_paths, output_file)
    print(f"Grouped bar chart saved to {output_file}")

if __name__ == "__main__":
    main()
