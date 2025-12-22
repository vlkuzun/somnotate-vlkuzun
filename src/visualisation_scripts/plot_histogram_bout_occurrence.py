import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_for_somno_QM_checks import get_bout_durations

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

def compare_histograms(file_1, file_2, sampling_rate=1, max_duration=600, bin_size=60, output_path=None):
    """
    Compares histograms of bout durations between two CSV files with customizable bins,
    and plots histogram bars in different colors for each file.

    Input:
        file_1 (str): Path to the first CSV file
        file_2 (str): Path to the second CSV file
        sampling_rate (int): Sampling rate of the data in Hz (default is 1Hz)
        max_duration (int): Maximum bout duration to include in seconds (default is 600)
        bin_size (int): Size of histogram bins in seconds (default is 60)
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

    # Restrict data to only include bout durations up to max_duration seconds
    bout_durations_1 = bout_durations_1[bout_durations_1 <= max_duration]
    bout_durations_2 = bout_durations_2[bout_durations_2 <= max_duration]

    # Set up histogram parameters with customizable bin size
    bins = np.arange(0, max_duration + bin_size, bin_size)

    # Plotting histograms - use rcParams for figure size
    plt.figure()

    # Plot histogram for file 1
    plt.hist(bout_durations_1, bins=bins, alpha=0.7, label="Somnotate", color='#1f77b4', edgecolor='black', linewidth=0.5)

    # Plot histogram for file 2
    plt.hist(bout_durations_2, bins=bins, alpha=0.7, label="Manual", color="#41403e", edgecolor='black', linewidth=0.5)

    # Customize the plot using rcParams font sizes instead of hardcoded values
    plt.xlabel('Bout Epoch (seconds)')
    plt.ylabel('Occurrence')
    plt.title(f'{bin_size} second bouts')
    plt.legend()

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Use tight layout for better spacing
    plt.tight_layout()
    
    # Always save the figure if output_path is provided
    if output_path:
        # Remove any file extension from the output path to use as base name
        base_path = output_path.split('.')[0] if '.' in output_path else output_path
        
        # Add bin_size and max_duration to the output filename
        base_path = f"{base_path}_{bin_size}secbins_{max_duration}secs"
        
        # Save as PNG
        png_path = f"{base_path}.png"
        plt.savefig(png_path, bbox_inches='tight')
        print(f"Figure saved as PNG: {png_path}")
        
        # Save as PDF (vector format for publication)
        pdf_path = f"{base_path}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Figure saved as PDF: {pdf_path}")
        
        # Save as EPS format
        eps_path = f"{base_path}.eps"
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"Figure saved as EPS: {eps_path}")
    
    # Always show the plot regardless of whether it was saved
    plt.show()

# Define the output path for saving the figure (without extension and without bin/duration info)
output_file = "/Volumes/harris/volkan/somnotate-vlkuzun/plots/bout_duration/occurrence_bout_duration/francesca_histogram_occurrence_bout_duration_somno_vs_manual_sub-010"

compare_histograms(
    file_1="/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/automated_state_annotationoutput_sub-010_ses-01_recording-01_timestamped_sr-1hz.csv",
    file_2="/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/sub-010_ses-01_recording-01_data-sleepscore_fp_timestamped_sr-1hz.csv",
    sampling_rate=1,  # or other sampling rates if required
    max_duration=120,  # maximum bout duration to include (seconds)
    bin_size=15,  # size of histogram bins (seconds)
    output_path=output_file  # Pass the output path to save the figure
)