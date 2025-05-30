import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from brokenaxes import brokenaxes
from scipy import stats

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
    'figure.figsize': [8, 6],       # Width x Height in inches - standard publication size
    'axes.linewidth': 1,            # Thinner axis borders
    'pdf.fonttype': 42,             # Embed fonts properly in PDFs
    'ps.fonttype': 42
})

def analyze_bout_durations_light_dark_phase(file_subject_dict, output_folder):
    """
    Analyze bout durations from EEG data downsampled to 1Hz and create a combined plot with Light and Dark phases pooled for all files.

    Parameters:
        file_subject_dict (dict): Dictionary with file paths as keys and subject titles as values.
        output_folder (str): Path to the folder to save the plots.

    Returns:
        None
    """
    # DataFrame to store pooled results for all files
    pooled_data = []

    for file, subject_title in file_subject_dict.items():
        # Load the CSV file
        df = pd.read_csv(file)

        # Convert Timestamp column to datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Round the numbers in the sleepStage column to the nearest integer
        df['sleepStage'] = df['sleepStage'].round().astype(int)

        # Create a new column to track changes in sleep stage
        df['sleepStageChange'] = df['sleepStage'] != df['sleepStage'].shift()

        # Create a cumulative sum of changes to identify continuous instances
        df['boutId'] = df['sleepStageChange'].cumsum()

        # Determine the time period (light or dark) for each row
        def get_time_period(row):
            hour = row['Timestamp'].hour
            if 9 <= hour < 21:
                return 'Light'
            else:
                return 'Dark'

        df['timePeriod'] = df.apply(get_time_period, axis=1)

        # Group by boutId and calculate the count of rows for each bout
        bout_durations = df.groupby(['boutId']).size().reset_index(name='boutDuration')

        # Determine the time period for each bout based on the majority time
        bout_time_periods = df.groupby('boutId')['timePeriod'].apply(lambda x: x.value_counts().index[0]).reset_index()

        # Merge the bout durations with the time periods
        bout_durations = pd.merge(bout_durations, bout_time_periods, on='boutId')

        # Add the subject title as a column for identification
        bout_durations['subject'] = subject_title

        # Append to pooled data
        pooled_data.append(bout_durations)

    # Combine all data into a single DataFrame
    pooled_data = pd.concat(pooled_data, ignore_index=True)

    # Create the plot with broken axis - using rcParams for figure size
    fig = plt.figure()
    
    # Adjust the figure margins before creating the brokenaxes
    plt.subplots_adjust(bottom=0.25)
    
    # Create the brokenaxes plot on the pre-adjusted figure
    bax = brokenaxes(ylims=((0, 8400), (14000, pooled_data['boutDuration'].max()+100)), 
                    hspace=0.1, fig=fig)

    # Define unique x-tick positions for light and dark phases for each subject
    subjects = ['somnotate', 'fp', 'bh', 'vu']
    time_periods = ['Light', 'Dark']
    xticks = []
    xlabels = []

    # Modified color palette with grey decreasing in darkness:
    # 1st (somnotate Light): blue
    # 2nd (fp Light): darkest grey
    # 3rd (bh Light): medium grey 
    # 4th (vu Light): lightest grey
    # 5th (somnotate Dark): blue
    # 6th (fp Dark): darkest grey (same as 2nd)
    # 7th (bh Dark): medium grey (same as 3rd)
    # 8th (vu Dark): lightest grey (same as 4th)
    color_palette = ['#1f77b4', '#333333', '#777777', '#AAAAAA', '#1f77b4', '#333333', '#777777', '#AAAAAA']
    
    # Create color mapping for each subject-period combination
    color_map = {}
    idx = 0
    for period in time_periods:
        for subject in subjects:
            color_map[(subject, period)] = color_palette[idx]
            idx += 1

    # Explicitly calculate x positions before plotting
    x_positions = []
    for period in time_periods:
        for subject in subjects:
            x_positions.append(len(x_positions))

    # Reset plotting variables
    plot_index = 0

    for period in time_periods:
        for subject in subjects:
            # Extract data for the subject and time period
            subset = pooled_data[(pooled_data['subject'] == subject) & (pooled_data['timePeriod'] == period)]

            # Add scatter points with jitter for better visibility
            bax.scatter(
                np.random.normal(loc=plot_index, scale=0.15, size=len(subset)), # Jittered x-axis positions
                subset['boutDuration'], # Y-axis: bout duration
                alpha=0.6, s=10, label=f"{subject} ({period})" if period == 'Light' else None,
                color=color_map[(subject, period)]
            )

            # Store x-tick position and label
            xticks.append(plot_index)
            xlabels.append(f"{subject} ({period})")
            plot_index += 1

    # Set x-ticks and labels AFTER all plotting is complete
    bax.set_xticks(xticks)
    bax.set_xticklabels(['', 'somnotate (Light)', 'fp (Light)', 'bh (Light)', 'vu (Light)', 
                          'somnotate (Dark)', 'fp (Dark)', 'bh (Dark)', 'vu (Dark)'], 
                       rotation=45, ha='right')
    bax.set_ylabel('Bout Duration (seconds)', labelpad=35)

    # Remove top and right spines for cleaner plot
    for ax in bax.axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Save the combined plot as PNG and PDF
    combined_plot_filename = f"{output_folder}/all_subjects_combined_bout_durations_light_and_dark_updated.png"
    plt.savefig(combined_plot_filename, bbox_inches='tight')

    # Save as PDF format for high-quality vector graphics
    pdf_filename = f"{output_folder}/all_subjects_combined_bout_durations_light_and_dark_updated.pdf"
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    
    plt.show()
    plt.close()

    print("Plot saved successfully as PNG and PDF.")

    # Perform a t-test to check for statistical differences between Light and Dark phases
    light_data = pooled_data[pooled_data['timePeriod'] == 'Light']['boutDuration']
    dark_data = pooled_data[pooled_data['timePeriod'] == 'Dark']['boutDuration']
    t_stat, p_value = stats.ttest_ind(light_data, dark_data, equal_var=False)

    print(f"T-test results comparing pooled light vs dark across all subjects: t-statistic = {t_stat:.2f}, p-value = {p_value:.3g}")

    # Perform one-way ANOVA for Light and Dark groups separately
    for period in ['Light', 'Dark']:
        group_data = pooled_data[pooled_data['timePeriod'] == period]
        anova_result = stats.f_oneway(
            *[group['boutDuration'].values for _, group in group_data.groupby('subject')]
        )
        print(f"One-way ANOVA results for {period} group: F-statistic = {anova_result.statistic:.2f}, p-value = {anova_result.pvalue:.3g}")

# Example usage:
analyze_bout_durations_light_dark_phase(
    file_subject_dict={
        "/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/automated_state_annotationoutput_sub-010_ses-01_recording-01_timestamped_sr-1hz.csv": "somnotate",
        "/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/sub-010_ses-01_recording-01_data-sleepscore_fp_timestamped_sr-1hz.csv": "fp",
        "/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/sub-010_ses-01_recording-01_export(HBH)_timestamped_sr-1hz.csv": 'bh',
        "/Volumes/harris/volkan/somnotate-vlkuzun/somnotate_performance/sub-010_ses-01_recording-01_data-sleepscore_vu_timestamped_sr-1hz.csv": "vu"
    },
    output_folder="/Volumes/harris/volkan/somnotate-vlkuzun/plots/bout_duration/light_dark_bout_duration"
)
