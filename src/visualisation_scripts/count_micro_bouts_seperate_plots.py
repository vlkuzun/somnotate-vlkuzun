import pandas as pd
import matplotlib.pyplot as plt

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

# Create bar charts
def create_bar_charts(file_paths, output_file):
    """Create bar charts of micro bout quantities."""
    labels = ['Somnotate', 'fp', 'bh', 'vu']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # light blue, orange, green, red
    total_bouts = []
    wake_bouts = []
    non_rem_bouts = []
    rem_bouts = []
    
    for file_path in file_paths:
        df = load_data(file_path)
        micro_bouts = count_micro_bouts(df)
        total_bouts.append(sum(micro_bouts.values()))
        wake_bouts.append(micro_bouts[1])
        non_rem_bouts.append(micro_bouts[2])
        rem_bouts.append(micro_bouts[3])
    
    fig, axs = plt.subplots(2,2, figsize=(12, 10))
    
    axs[0, 0].bar(labels, total_bouts, color=colors)
    axs[0, 0].set_title('All Stages Micro Bouts (<15 seconds)')
    axs[0, 0].set_ylabel('Quantity')
    axs[0, 0].set_ylim(0, 45)
    
    axs[0, 1].bar(labels, wake_bouts, color=colors)
    axs[0, 1].set_title('Wake Micro Bouts (<15 seconds)')
    axs[0, 1].set_ylabel('Quantity')
    axs[0, 1].set_ylim(0, 45)
    
    axs[1, 0].bar(labels, non_rem_bouts, color=colors)
    axs[1, 0].set_title('Non-REM Micro Bouts (<15 seconds)')
    axs[1, 0].set_ylabel('Quantity')
    axs[1, 0].set_ylim(0, 45)
    
    axs[1, 1].bar(labels, rem_bouts, color=colors)
    axs[1, 1].set_title('REM Micro Bouts (<15 seconds)')
    axs[1, 1].set_ylabel('Quantity')
    axs[1, 1].set_ylim(0, 45)

    # Remove top and right spines for all subplots
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# Main function
def main():
    file_paths = [
        '/Volumes/harris/volkan/sleep_profile/francesca_sub-010/automated_state_annotationoutput_sub-010_ses-01_recording-01_sr-1hz.csv',
        '/Volumes/harris/volkan/sleep_profile/francesca_sub-010/sub-010_ses-01_recording-01_data-sleepscore_fp_sr-1hz.csv',
        '/Volumes/harris/volkan/sleep_profile/francesca_sub-010/sub-010_ses-01_recording-01_data-sleepscore_vu_sr-1hz.csv',
        '/Volumes/harris/volkan/sleep_profile/francesca_sub-010/sub-010_ses-01_recording-01_export(HBH)_sr-1hz.csv'

    ]
    output_file = '/Volumes/harris/volkan/sleep_profile/figures/somnotate_control/micro_bout_bar_charts.png'
    create_bar_charts(file_paths, output_file)
    print(f"Bar charts saved to {output_file}")

if __name__ == "__main__":
    main()