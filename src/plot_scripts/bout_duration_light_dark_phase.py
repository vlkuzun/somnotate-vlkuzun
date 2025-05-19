import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

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

    # Prepare data for plotting
    plt.figure(figsize=(14, 10))

    # Create a strip plot for individual data points
    sns.stripplot(data=pooled_data, x='timePeriod', y='boutDuration', jitter=True, dodge=True, hue='subject', alpha=0.6)

    # Customize the plot
    plt.xlabel('')
    plt.ylabel('Bout Duration (seconds)', fontsize=14)
    plt.title('Somnotate Performance - Bout Durations for Light and Dark Phases', fontsize=18)
    plt.xticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left', title="Subject")

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the combined plot
    combined_plot_filename = f"{output_folder}/all_subjects_combined_bout_durations.png"
    plt.savefig(combined_plot_filename, bbox_inches='tight')
    plt.close()

    print("Combined analysis and plot saved successfully.")

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
        "/Volumes/harris/volkan/somnotate/somnotate_performance/automated_state_annotationoutput_sub-010_ses-01_recording-01_timestamped_sr-1hz.csv": "somnotate",
        "/Volumes/harris/volkan/somnotate/somnotate_performance/sub-010_ses-01_recording-01_data-sleepscore_fp_timestamped_sr-1hz.csv": "fp",
        "/Volumes/harris/volkan/somnotate/somnotate_performance/sub-010_ses-01_recording-01_export(HBH)_timestamped_sr-1hz.csv": 'bh',
        "/Volumes/harris/volkan/somnotate/somnotate_performance/sub-010_ses-01_recording-01_data-sleepscore_vu_timestamped_sr-1hz.csv": "vu"
    },

    output_folder="/Volumes/harris/volkan/somnotate/figures/bout_duration"
)