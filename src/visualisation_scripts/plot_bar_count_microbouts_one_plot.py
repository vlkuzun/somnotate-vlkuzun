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
    
    # Use consistent color scheme with alpha=0.7
    colors = ['#1f77b4', '#41403e']  # Blue (Somnotate) and Dark Gray (Manual)

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

    # Create the plot - using rcParams for figure size
    fig, ax = plt.subplots()
    
    bar_width = 0.35
    for i, (label, color) in enumerate(zip(labels, colors)):
        values = [data[stage][i] for stage in stages]
        ax.bar([pos + i * bar_width for pos in x], values, bar_width, 
               label=label, color=color, alpha=0.7, 
               edgecolor='black', linewidth=0.5)  # Add black edge with 0.5 width

    # Formatting the plot using rcParams for font sizes
    ax.set_xticks([pos + bar_width / 2 for pos in x])
    ax.set_xticklabels(['Wake', 'NREM', 'REM'])
    ax.set_ylabel('Occurrence')
    
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save plot in PNG and PDF formats
    if output_file:
        # Extract base path without extension
        base_output = output_file.split('.')[0] if '.' in output_file else output_file
        
        # Save as PNG
        png_path = f"{base_output}.png"
        plt.savefig(png_path, bbox_inches='tight')
        print(f"Figure saved as PNG: {png_path}")
        
        # Save as PDF (vector format for publication)
        pdf_path = f"{base_output}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved as PDF: {pdf_path}")
    
    # Display the plot
    plt.show()
    plt.close()

# Main function
def main():
    file_paths = [
        '/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/automated_state_annotationoutput_sub-010_ses-01_recording-01_timestamped_sr-1hz.csv',
        '/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/sub-010_ses-01_recording-01_data-sleepscore_fp_timestamped_sr-1hz.csv',
        '/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/sub-010_ses-01_recording-01_data-sleepscore_vu_timestamped_sr-1hz.csv',
        '/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/sub-010_ses-01_recording-01_export(HBH)_timestamped_sr-1hz.csv'
    ]
    output_file = '/Volumes/harris/volkan/somnotate-vlkuzun/plots/bout_duration/microbouts/micro_bout_bar_charts_one_plot'
    create_grouped_bar_chart(file_paths, output_file)
    print(f"Grouped bar chart saved to {output_file}")

if __name__ == "__main__":
    main()
