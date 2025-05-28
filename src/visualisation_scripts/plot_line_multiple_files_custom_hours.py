import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle

from scipy.ndimage import gaussian_filter1d

def plot_sleep_stages(files, subjects, fontsize=24):
    """
    Plots wake_percent, non_rem_percent, and rem_percent from multiple files on the same graph.
    Adds a 12-hour cycle bar beneath the main graph, colored based on ZT_Adjusted values.

    Args:
        files (list of str): List of file paths. Each file should have columns ZT, wake_percent, non_rem_percent, and rem_percent.
        subjects (list of str): List of subject names corresponding to each file.
        fontsize (int): Base font size for plot elements (default: 24).
    """
    if not isinstance(files, list) or len(files) == 0:
        raise ValueError("Please provide a list of file paths.")

    if not isinstance(subjects, list) or len(subjects) != len(files):
        raise ValueError("Please provide a list of subject names matching the number of files.")

    # Define consistent colors for each sleep stage
    colors = {
        'wake_percent': 'red',
        'non_rem_percent': 'blue',
        'rem_percent': 'green',
    }

    # Define line styles for different files
    line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

    # Map stage names to user-friendly labels
    stage_labels = {
        'wake_percent': 'Awake',
        'non_rem_percent': 'NREM',
        'rem_percent': 'REM'
    }

    # Set the font sizes globally
    plt.rcParams.update({'font.size': fontsize})
    
    # Create the main plot with more space at the top for legend
    fig = plt.figure(figsize=(16, 4))  # Increased figure height to make room for legend
    
    # Create subplot with adjusted position to leave room at the top
    ax1 = fig.add_subplot(111)

    min_zt_adjusted = float('inf')
    max_zt_adjusted = -float('inf')


    for idx, (file, subject) in enumerate(zip(files, subjects)):
        # Read the file into a DataFrame
        try:
            data = pd.read_csv(file)
        except Exception as e:
            print(f"Could not read file {file}: {e}")
            continue

        # Check if required columns exist
        required_columns = ['ZT', 'wake_percent', 'non_rem_percent', 'rem_percent']
        if not all(col in data.columns for col in required_columns):
            print(f"File {file} is missing required columns.")
            continue

        # Adjust ZT to create unique x-axis values for continuous cycles
        max_zt = 24
        unique_x_values = []
        last_zt = None
        cycle = 0

        for zt in data['ZT']:
            if last_zt is not None and zt < last_zt:
                cycle += 1
            unique_x_values.append(zt + cycle * max_zt)
            last_zt = zt

        data['ZT_Adjusted'] = unique_x_values

        # Update min and max ZT_Adjusted if data exists
        if not data['ZT_Adjusted'].empty:
            min_zt_adjusted = min(min_zt_adjusted, data['ZT_Adjusted'].min())
            max_zt_adjusted = max(max_zt_adjusted, data['ZT_Adjusted'].max())

        # Smooth the data with Gaussian filter (sigma=1)
        smoothed_data = {
            stage: gaussian_filter1d(data[stage], sigma=1)
            for stage in ['wake_percent', 'non_rem_percent', 'rem_percent']
        }

        # Identify gaps in the ZT sequence and insert NaN
        adjusted = [data['ZT_Adjusted'].iloc[0]]
        for i in range(1, len(data)):
            if data['ZT_Adjusted'].iloc[i] - data['ZT_Adjusted'].iloc[i - 1] > 1:
                # Insert NaN for the gap
                adjusted.append(np.nan)
            adjusted.append(data['ZT_Adjusted'].iloc[i])

        for stage in ['wake_percent', 'non_rem_percent', 'rem_percent']:
            values = smoothed_data[stage].tolist()
            adjusted_values = [values[0]]
            for i in range(1, len(values)):
                if data['ZT_Adjusted'].iloc[i] - data['ZT_Adjusted'].iloc[i - 1] > 1:
                    # Insert NaN for the gap
                    adjusted_values.append(np.nan)
                adjusted_values.append(values[i])

            # Plot the adjusted data
            ax1.plot(adjusted, adjusted_values, linestyle=line_styles[idx % len(line_styles)], 
                     color=colors[stage], linewidth=1, label=f"{stage_labels[stage]} ({subject})")

    # Handle case where no valid data exists
    if max_zt_adjusted == -float('inf') or min_zt_adjusted == float('inf'):
        print("No valid data found to plot.")
        return

    # Add a 12-hour cycle bar directly below the graph using rectangles
    total_cycles = int(np.ceil(max_zt_adjusted / 12))  # Number of full 12-hour cycles
    y_min, y_max = ax1.get_ylim()
    bar_height = (y_max - y_min) * 0.05  # Height of the bar as 5% of the y-axis range
    bar_y_start = y_min - bar_height  # Position the bar just below the visible range

    for i in range(total_cycles):
        start = i * 12
        end = (i + 1) * 12
        color = 'orange' if i % 2 == 0 else 'grey'

        # Add the rectangle for the cycle bar
        ax1.add_patch(Rectangle((start, bar_y_start), width=end - start, height=bar_height, color=color, alpha=0.6))

    # Adjust the y-axis limits to include the cycle bar
    ax1.set_ylim(bar_y_start, y_max)

    # Adjust the x-axis to cover the full range of ZT_Adjusted
    ax1.set_xlim(min_zt_adjusted, max_zt_adjusted)

     # Find the first occurrence of ZT=12
    first_zt_12_idx = data[data['ZT'] == 12].index[0] if not data[data['ZT'] == 12].empty else None
    if first_zt_12_idx is None:
        raise ValueError("ZT=12 not found in the data. Please check the input data.")
    
    # Generate tick locations and labels for every ZT=0 or ZT=12
    xtick_locations = data.loc[data['ZT'].isin([0, 12]), 'ZT_Adjusted'].values
    xtick_labels = [int(label) for label in data.loc[data['ZT'].isin([0, 12]), 'ZT'].values]

    # Set x-axis ticks and labels
    ax1.set_xticks(xtick_locations)
    ax1.set_xticklabels(xtick_labels, fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)

    # Adding labels, legend, and title to the main plot
    ax1.set_xlabel('Zeitgeber Time', fontsize=fontsize+2)
    ax1.set_ylabel('Percentage', fontsize=fontsize+2)
    
    # Create custom legend items to organize by subject and ensure row-wise filling
    from matplotlib.lines import Line2D
    
    # Manually define the legend entries
    legend_elements = [
        Line2D([0], [0], color=colors['wake_percent'], linestyle=line_styles[0],
               label="Wake (Somnotate)"),
        Line2D([0], [0], color=colors['wake_percent'], linestyle=line_styles[1],
               label="Wake (Manual)"),
        Line2D([0], [0], color=colors['non_rem_percent'], linestyle=line_styles[0],
               label="NREM (Somnotate)"),
        Line2D([0], [0], color=colors['non_rem_percent'], linestyle=line_styles[1],
               label="NREM (Manual)"),
        Line2D([0], [0], color=colors['rem_percent'], linestyle=line_styles[0],
               label="REM (Somnotate)"),
        Line2D([0], [0], color=colors['rem_percent'], linestyle=line_styles[1],
               label="REM (Manual)")
    ]

    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.55),
               ncol=3, fontsize=fontsize-6, frameon=True, 
               handlelength=1.5, columnspacing=1.0, 
               handletextpad=0.5)
    
    # Remove top and right spines for cleaner plot
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Make the remaining spines thicker for better visibility
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['left'].set_linewidth(1.5)
    
    # Also make the tick marks thicker
    ax1.tick_params(width=1.5, length=6)

    # Set x-axis ticks at 0, 50, 100
    ax1.set_yticks([0, 50, 100])
    ax1.set_yticklabels(['0', '50', '100'], fontsize=fontsize)
   
    # Use subplots_adjust instead of tight_layout to have more control
    # This creates more space at the top for the legend
    plt.subplots_adjust(top=0.65, bottom=0.25, left=0.1, right=0.95)

    # Save figure in both PNG and EPS formats
    output_path = '/Volumes/harris/volkan/somnotate-vlkuzun/plots/stage_across_ZT/somnotate_vs_manual_sub-010_fp_stage_across_ZT_70hr'
    plt.savefig(f"{output_path}.png", dpi=600)
    plt.savefig(f"{output_path}.eps", format='eps')
    print(f"Figure saved as {output_path}.png with 600 DPI and as {output_path}.eps")

    # Show the plot
    plt.show()

# Example usage
plot_sleep_stages(['/Volumes/harris/volkan/sleep-profile/downsample_auto_score/sub-010_ses-01_recording-01_time-0-69h_1Hz_1hrbins.csv', 
                  '/Volumes/harris/volkan/sleep-profile/downsample_manual_score/sub-010_ses-01_recording-01_time-0-69h_manual_sr-1hz_1hrbins_ZT.csv'], 
                 ['Somnotate', 'Manual'],
                 fontsize=24)
