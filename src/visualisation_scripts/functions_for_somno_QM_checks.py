import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def match_length_csv_files(df1, df2):
    '''
    Check if the two CSV files have the same number of samples. If not, truncate the longer file to match the length of the shorter file.
    Input:
        df1: Path to the first CSV file
        df2: Path to the second CSV file
    Output:
        df1: Truncated CSV file 1
        df2: Truncated CSV file 2
    '''
    
    len_csv1 = len(df1)
    len_csv2 = len(df2)

    if len_csv1 != len_csv2:
        print(f"Length mismatch: CSV1 has {len_csv1} samples, CSV2 has {len_csv2} samples.")
        if len_csv1 > len_csv2:
            excess_rows = len_csv1 - len_csv2
            df1 = df1[:-excess_rows]
            print(f"CSV1 truncated by {excess_rows} samples to match length of CSV2")
        else:
            excess_rows = len_csv2 - len_csv1
            df2 = df2[:-excess_rows]
            print(f"CSV2 truncated by {excess_rows} samples to match length of CSV1 ") 
    
    assert len(df1) == len(df2), "Length of CSV1 does not match length of CSV2 after truncation"

    return df1, df2


def compare_csv_files(df1, df2):
    ''' 
    Compare the sleep stages from two CSV files.
    Input:
        df1: Path to the first CSV file
        df2: Path to the second CSV file
    Output:
        percentage_similarity: Percentage of samples where the sleep stages match between the two CSV files
    '''

    matches = df1['sleepStage'] == df2['sleepStage'] # element-wise comparison of the two columns
    percentage_similarity = np.mean(matches) * 100
    print(f"Percentage similarity: {percentage_similarity:.2f}%")

    return percentage_similarity

def rename_file(file_path):
    '''
    Label the files based on their filenames.
    Input:
        file_path: Path to the file
    Output:
        label: Label for the file
    '''
    filename = os.path.basename(file_path)
    if "automated" in filename:
        return "somnotate"
    if "fp" in filename:
        return "fp"
    if "vu" in filename:
        return "vu"
    if "BH" in filename:
        return "bh"
    else:
        return "control"

def compare_csv_files_by_stage(df_manual, df_somnotate, stage_value):
    ''' 
    Compare specific sleep stages between a manual CSV file and the somnotate CSV file.
    Input:
        df_manual: DataFrame for the manual file
        df_somnotate: DataFrame for the somnotate file
        stage_value: The value of the sleep stage to filter by (1: awake, 2: non-REM, 3: REM, etc.)
    Output:
        percentage_similarity: Percentage similarity between the manual and somnotate annotations for this sleep stage
    '''

    # Get indices in the manual annotations where the sleep stage is equal to a given stage_value (e.g., 1 for 'awake')
    manual_stage_indices = df_manual[df_manual['sleepStage'] == stage_value].index

    # Get the corresponding sleep stages in somnotate based on these indices
    somnotate_stage_at_indices = df_somnotate.loc[manual_stage_indices, 'sleepStage']

    # Get the manual sleep stages at those same indices (should all be stage_value)
    manual_stage_at_indices = df_manual.loc[manual_stage_indices, 'sleepStage']
    assert manual_stage_at_indices.all() == stage_value, f"Manual sleep stages at indices {manual_stage_indices} are not all {stage_value}"

    # Compare the two (manual vs somnotate) for those indices
    matches = manual_stage_at_indices == somnotate_stage_at_indices
    percentage_similarity = np.mean(matches) * 100

    print(f"Percentage similarity for sleep stage {stage_value} (manual vs somnotate): {percentage_similarity:.2f}%")
    
    return percentage_similarity

def compute_confusion_matrix_by_stage(df_manual, df_somnotate, stages):
    ''' 
    Compute confusion matrix for misclassification of sleep stages, after checking for length mismatch.
    Input:
        df_manual: DataFrame for the manual file
        df_somnotate: DataFrame for the somnotate file
        stages: Dictionary mapping sleep stage names to their respective values (e.g., {"awake": 1, "non-REM": 2, "REM": 3})
    Output:
        confusion_matrix: A matrix with counts of misclassifications between stages
    '''
    # Ensure both DataFrames are of equal length
    df_manual, df_somnotate = match_length_csv_files(df_manual, df_somnotate) 

    # Initialize the confusion matrix (N x N, where N is the number of stages)
    num_stages = len(stages)
    confusion_matrix = np.zeros((num_stages, num_stages))  # Rows: manual, Columns: somnotate

    for manual_stage_name, manual_stage_value in stages.items():

        manual_stage_indices = df_manual[df_manual['sleepStage'] == manual_stage_value].index # get the indices where the manual stage is equal to the current stage value (e.g., 'awake')
        somnotate_stage_at_indices = df_somnotate.loc[manual_stage_indices, 'sleepStage'] # get the somnotate stage at the same indices

        for somnotate_stage_value in somnotate_stage_at_indices:
            confusion_matrix[stages[manual_stage_name] - 1, somnotate_stage_value - 1] += 1

    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, labels, title="Sleep Stage Confusion Matrix"):
    '''
    Plot the confusion matrix showing misclassifications between sleep stages.
    Input:
        confusion_matrix: A NxN matrix with misclassification counts
        labels: List of sleep stages (e.g., ['awake', 'non-REM', 'REM'])
        title: Title for the plot
    '''
    # Normalize the confusion matrix by dividing by row sums to get percentages
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = confusion_matrix / row_sums

    # Create a DataFrame for the heatmap
    df = pd.DataFrame(normalized_matrix, index=labels, columns=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues", cbar=True, linewidths=0.5)
    plt.title(title)
    plt.xlabel('Somnotate Annotation')
    plt.ylabel('Manual Annotation')
    plt.tight_layout()
    plt.show()


def get_bout_durations(df, sampling_rate, df_name):
    '''
    Calculate the duration of bouts for each sleep stage in a CSV file.
    Input:
        df: DataFrame for the CSV file
        sampling_rate: Sampling rate of the data in Hz
    Output:
        bout_durations_with_stage_all: Dictionary with DataFrame for each CSV file containing bout durations and corresponding sleep stages
    '''
        
    print(f"Type of input df: {type(df)}")  # Debug: Check type of df

    # if 'sleepStage' not in df.columns:
    #     raise ValueError("The 'sleepStage' column is missing from the DataFrame.")
    
    bout_durations_with_stage = []
    bout_durations = []
    bout_stages = []
    bout_durations_with_stage_all = {}

    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]

    previous_time = 0 
    previous_stage = df['sleepStage'].iloc[0]
    

    for stage_change in stage_changes:
        bout_duration = (stage_change - previous_time) / sampling_rate
        bout_durations.append(bout_duration)
        bout_stages.append(previous_stage)

        previous_time = stage_change
        previous_stage = df['sleepStage'].iloc[stage_change]

    final_bout_duration = (len(df) - previous_time) / sampling_rate
    bout_durations.append(final_bout_duration)  # Add the duration of the last bout
    bout_stages.append(previous_stage) # Add the stage of the last bout

    bout_durations_with_stage = pd.DataFrame({'BoutDuration': bout_durations, 'SleepStage': bout_stages})
    bout_durations_with_stage_all[df_name] = bout_durations_with_stage

    return bout_durations_with_stage_all

def get_stage_durations(bout_durations_with_stage_all):
    '''
    Extract bout durations for awake, NREM, and REM stages from all dataframes.
    Input:
        bout_durations_with_stage_all: Dictionary of DataFrames containing bout durations and sleep stages
    Output:
        bout_durations_awake, bout_durations_nrem, bout_durations_rem: Dictionaries of bout durations for each stage
    '''
    
    bout_durations_awake = {}
    bout_durations_nrem = {}
    bout_durations_rem = {}

    for df_name, df in bout_durations_with_stage_all.items():
        # Extract bout durations for specific sleep stages (1: awake, 2: NREM, 3: REM)
        awake_durations = df.loc[df['SleepStage'] == 1, 'BoutDuration'].tolist()
        nrem_durations = df.loc[df['SleepStage'] == 2, 'BoutDuration'].tolist()
        rem_durations = df.loc[df['SleepStage'] == 3, 'BoutDuration'].tolist()

        bout_durations_awake[df_name] = awake_durations
        bout_durations_nrem[df_name] = nrem_durations
        bout_durations_rem[df_name] = rem_durations

    return bout_durations_awake, bout_durations_nrem, bout_durations_rem

def perform_anova(bout_durations_dict, sleep_stage_label):

    if sleep_stage_label == 'All Stages':
        print("No sleep stage label provided. Performing ANOVA on all sleep stages combined.")
        # Extract all bout durations from the dictionaries for each dataframe
        all_durations = {df_name: df['BoutDuration'].tolist() for df_name, df in bout_durations_dict.items()}
        # Convert the dictionary into a list of lists (each list corresponds to bout durations from one dataframe)
        data = [durations for durations in all_durations.values()]

    else:
        print(f"Performing ANOVA for {sleep_stage_label}")
        data = [durations for durations in bout_durations_dict.values()]
    
    # Perform one-way ANOVA
    f_stat, p_value = stats.f_oneway(*data)
    
    print(f"ANOVA results for {sleep_stage_label}:")
    print(f"F-statistic: {f_stat}")
    print(f"P-value: {p_value}")
    
    # Interpretation
    if p_value < 0.05:
        print(f"Significant differences found between dataframes for {sleep_stage_label} (p < 0.05)\n")
    
    return f_stat, p_value

def tukey_test(bout_durations_dict, sleep_stage_label):
    '''
    Perform Tukey's post-hoc test to compare the means of bout durations between different dataframes.
    Input:
        bout_durations_dict: Dictionary of bout durations for each dataframe

    Output:
        tukey_results: Results of the Tukey's HSD test
    '''

    # Combine all bout durations into a single list and add labels for dataframes
    durations = []
    labels = []
    
    for df_name, duration_list in bout_durations_dict.items():
        if sleep_stage_label == 'All Stages':
            if isinstance(duration_list, pd.DataFrame):
                duration_list = duration_list['BoutDuration'].tolist()
            else:
                raise ValueError(f"Expected DataFrame for {df_name} when sleep_stage_label is None, got {type(duration_list)}")
        else:
            duration_list = duration_list    
        
        durations.extend(duration_list)
        labels.extend([df_name] * len(duration_list))
    
    
    # Perform Tukey's HSD
    data = pd.DataFrame({'BoutDuration': durations, 'DataFrame': labels})
    tukey = pairwise_tukeyhsd(endog=data['BoutDuration'], groups=data['DataFrame'], alpha=0.05)
    print(tukey)
    
    return tukey       

def remove_outliers(data):
    """
    Removes outliers based on the IQR method.
    
    Parameters:
        data (list/array): Data to check for outliers.
        
    Returns:
        filtered_data (list): Data without outliers.
    """
    # Calculate quartiles
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter data points within bounds
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    
    return filtered_data

def plot_bout_duration_histograms_with_significance(bout_durations_dict, sleep_stage_label):
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
    
    plt.figure()
    labels = list(bout_durations_dict.keys())
    
    # Calculate means and standard errors
    means = []
    ses = []
    for df_name in labels:
        data = bout_durations_dict[df_name]
        if sleep_stage_label == 'All Stages':
                data = data['BoutDuration'].tolist()
        
        means.append(np.mean(data))
        ses.append(stats.sem(data) if len(data) > 1 else 0)  # SEM, avoid division by zero

    x = np.arange(len(labels))  # X-axis positions for bars
    y_max = max(means) + max(ses)  # Max y value for positioning brackets
    
    # Plot bars with error bars (means and standard errors)
    plt.bar(x, means, yerr=ses, capsize=5, color=['#1f77b4', '#333333', '#777777', '#AAAAAA'], alpha=0.7)
    plt.xticks(x, labels)
    plt.ylabel('Bout Duration (seconds)')
    
    plt.ylim(0, y_max + 0.2 * y_max)  # Adjust y-limits

    # Perform Tukey's HSD if multiple dataframes are available
    if len(labels) > 1:
        tukey = tukey_test(bout_durations_dict, sleep_stage_label)
        comparisons = tukey._results_table.data[1:]  # Extract results from Tukey's test
        significance_threshold = 0.05  # Significance level for stars

        # Annotate significance stars on pairwise comparisons
        for comparison in comparisons:
            group1, group2, meandiff, p_adj, lower, upper, reject = comparison
            if p_adj < significance_threshold:  # Only annotate significant differences
                idx1 = labels.index(group1)
                idx2 = labels.index(group2)
                label_position = (idx1 + idx2) / 2  # Position between the two bars

                # Calculate y_position for the bracket and significance label
                y_position = y_max + 0.1 * y_max
                plt.text(label_position, y_position, '*', ha='center', va='bottom')

                # Add brackets
                plt.plot([idx1, idx1, idx2, idx2], [y_max, y_position, y_position, y_max], lw=1.5, color='black')

    f_stat, p_value = perform_anova(bout_durations_dict, sleep_stage_label)

    # Remove top and right spines for cleaner plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust spacing to fit the title and plot comfortably
    plt.tight_layout()
    
    # Save the figure in both PNG and PDF formats
    base_filename = f"/Volumes/harris/volkan/somnotate-vlkuzun/plots/bout_duration/barplot_bout_duration/barplot_bout_duration_{sleep_stage_label.replace(' ', '_')}"
    
    # Save as PNG
    plt.savefig(f"{base_filename}.png", bbox_inches='tight')
    
    # Save as PDF for publication quality
    plt.savefig(f"{base_filename}.pdf", format='pdf', bbox_inches='tight')
    
    plt.show()
    plt.close()

def plot_bout_duration_barplot_stripplot_with_significance(bout_durations_dict, sleep_stage_label):
    plt.figure(figsize=(14, 8))
    labels = list(bout_durations_dict.keys())
    
    # Calculate means and standard errors
    means = []
    ses = []
    for df_name in labels:
        data = bout_durations_dict[df_name]
        if sleep_stage_label == 'All Stages':
            data = data['BoutDuration'].tolist()
        
        means.append(np.mean(data))
        ses.append(stats.sem(data) if len(data) > 1 else 0)  # SEM, avoid division by zero

    x = np.arange(len(labels))  # X-axis positions for bars
    y_max = max(means) + max(ses)  # Max y value for positioning brackets

    # Plot bars with error bars (means and standard errors)
    plt.bar(x, means, yerr=ses, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)

    # Overlay individual data points as a stripplot
    for i, df_name in enumerate(labels):
        data = bout_durations_dict[df_name]
        if sleep_stage_label == 'All Stages':
            data = data['BoutDuration'].tolist()
        
        sns.stripplot(
            x=[i] * len(data),  # Position for each group
            y=data, 
            color='black', 
            alpha=0.5, 
            jitter=True,
            size=5
        )

    plt.xticks(x, labels)
    plt.ylabel('Mean Bout Duration (seconds)')

    if sleep_stage_label:
        plt.title(f'Bout Durations for {sleep_stage_label}', fontsize=16, pad=20)  # Add padding
    else:
        plt.title('Overall Bout Durations Across DataFrames', fontsize=16, pad=20)

    # Increase y-limits to make room for annotations (brackets and stars)
    additional_space = 0.6  # Extra space factor above the maximum bar
    plt.ylim(0, y_max + additional_space * y_max)

    # Perform Tukey's HSD if multiple dataframes are available
    if len(labels) > 1:
        tukey = tukey_test(bout_durations_dict, sleep_stage_label)
        comparisons = tukey._results_table.data[1:]  # Extract results from Tukey's test
        significance_threshold = 0.05  # Significance level for stars

        # Annotate significance stars on pairwise comparisons
        for comparison in comparisons:
            group1, group2, meandiff, p_adj, lower, upper, reject = comparison
            if p_adj < significance_threshold:  # Only annotate significant differences
                idx1 = labels.index(group1)
                idx2 = labels.index(group2)
                label_position = (idx1 + idx2) / 2  # Position between the two bars

                y_position = y_max + 0.4 * y_max  # Adjusting bracket position
                star_position = y_position + 0.1 * y_max  # Position asterisk slightly above the bracket

                plt.text(label_position, star_position, '*', ha='center', va='bottom', fontsize=12)

                # Add brackets closer to the asterisk
                plt.plot([idx1, idx1, idx2, idx2], 
                         [y_position, star_position - 0.05 * y_max, star_position - 0.05 * y_max, y_position], 
                         lw=1.5, color='black')

    f_stat, p_value = perform_anova(bout_durations_dict, sleep_stage_label)

    # Remove top and right spines for cleaner plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust spacing to fit the title and plot comfortably
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to prevent overlap with title

    plt.show()


def plot_bout_duration_barplot_stripplot_with_significance_all_data(bout_durations_dict, sleep_stage_label):
    plt.figure(figsize=(14, 8))
    labels = list(bout_durations_dict.keys())
    
    # Calculate means and standard errors
    means = []
    ses = []
    all_values = []  # Collect all values to find the maximum
    for df_name in labels:
        data = bout_durations_dict[df_name]
        if sleep_stage_label == 'All Stages':
            data = data['BoutDuration'].tolist()
        
        means.append(np.mean(data))
        ses.append(stats.sem(data) if len(data) > 1 else 0)  # SEM, avoid division by zero
        all_values.extend(data)  # Extend the list with all values

    x = np.arange(len(labels))  # X-axis positions for bars
    dataset_max = max(all_values)  # Find the maximum value in the dataset
    y_max = max(means) + max(ses)  # Max y value for positioning brackets

    # Plot bars with error bars (means and standard errors)
    plt.bar(x, means, yerr=ses, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)

    # Overlay individual data points as a stripplot
    for i, df_name in enumerate(labels):
        data = bout_durations_dict[df_name]
        if sleep_stage_label == 'All Stages':
            data = data['BoutDuration'].tolist()
        
        sns.stripplot(
            x=[i] * len(data),  # Position for each group
            y=data, 
            color='black', 
            alpha=0.5, 
            jitter=True,
            size=5
        )

    plt.xticks(x, labels, fontsize=14)
    plt.ylabel('Bout Duration (seconds)', fontsize=14)

    if sleep_stage_label:
        plt.title(f'Somnotate Performance - Bout Durations Across {sleep_stage_label}', fontsize=18, pad=30)  # Add padding
    else:
        plt.title('Overall Bout Durations Across DataFrames', fontsize=18, pad=30)

    # Increase y-limits to make room for annotations (brackets and stars)
    additional_space = 0.1  # Extra space factor above the dataset max
    plt.ylim(0, dataset_max + additional_space * dataset_max)

    # Perform Tukey's HSD if multiple dataframes are available
    if len(labels) > 1:
        tukey = tukey_test(bout_durations_dict, sleep_stage_label)
        comparisons = tukey._results_table.data[1:]  # Extract results from Tukey's test
        significance_threshold = 0.05  # Significance level for stars

        # Annotate significance stars on pairwise comparisons
        for comparison in comparisons:
            group1, group2, meandiff, p_adj, lower, upper, reject = comparison
            if p_adj < significance_threshold:  # Only annotate significant differences
                idx1 = labels.index(group1)
                idx2 = labels.index(group2)
                label_position = (idx1 + idx2) / 2  # Position between the two bars

                y_position = dataset_max - 5  # Bracket position above dataset max
                star_position = y_position + 0.05 * y_position  # Asterisk position slightly above the bracket

                plt.text(label_position, star_position, '*', ha='center', va='bottom', fontsize=12)

                # Add brackets closer to the asterisk
                plt.plot([idx1, idx1, idx2, idx2], 
                         [y_position, y_position + 0.05 * y_position, y_position + 0.05 * y_position, y_position], 
                         lw=1.5, color='black')

    f_stat, p_value = perform_anova(bout_durations_dict, sleep_stage_label)

    # Remove top and right spines for cleaner plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust spacing to fit the title and plot comfortably
    plt.tight_layout()  # Adjust to prevent overlap with title

    plt.show()

def plot_bout_duration_barplot_stripplot_with_significance_all_data_remove_outlier(bout_durations_dict, sleep_stage_label):
    """
    Plots a barplot with means and SEMs, along with a stripplot overlaid showing individual data points,
    excluding outliers only in the stripplot visualization. Y-axis limits depend on the filtered distribution.
    
    Parameters:
        bout_durations_dict (dict): Dictionary where keys are subject names and values are DataFrames containing BoutDuration data.
        sleep_stage_label (str): Label for the sleep stages (e.g., 'All Stages') for the analysis.
    """
    plt.figure(figsize=(14, 8))
    labels = list(bout_durations_dict.keys())
    
    # Calculate means and standard errors
    means = []
    ses = []
    all_filtered_values = []  # Collect values after outlier removal for determining y-axis limits
    for df_name in labels:
        data = bout_durations_dict[df_name]
        if sleep_stage_label == 'All Stages':
            data = data['BoutDuration'].tolist()

        # Remove outliers for the stripplot calculation
        filtered_data = remove_outliers(data)

        means.append(np.mean(data))  # Mean using original data
        ses.append(stats.sem(data) if len(data) > 1 else 0)  # SEM based on original data
        all_filtered_values.extend(filtered_data)  # Extend with filtered data for y-axis limits

    x = np.arange(len(labels))  # X-axis positions for bars

    # Dynamically set y-axis limits based on filtered stripplot data
    y_max = max(all_filtered_values) + 0.1 * max(all_filtered_values)  # 10% padding
    y_min = min(all_filtered_values) - 0.1 * max(all_filtered_values)  # 10% padding

    # Plot bars with error bars (means and standard errors)
    plt.bar(x, means, yerr=ses, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)

    # Overlay individual data points as a stripplot with outlier removal
    for i, df_name in enumerate(labels):
        data = bout_durations_dict[df_name]
        if sleep_stage_label == 'All Stages':
            data = data['BoutDuration'].tolist()

        # Remove outliers only for the stripplot
        filtered_data = remove_outliers(data)

        # Plot the filtered stripplot
        sns.stripplot(
            x=[i] * len(filtered_data),  # Position for each group
            y=filtered_data, 
            color='black', 
            alpha=0.5, 
            jitter=True,
            size=5
        )

    plt.xticks(x, labels, fontsize=14)
    plt.ylabel('Bout Duration (seconds)', fontsize=14)

    if sleep_stage_label:
        plt.title(f'Somnotate Bout Duration Comparison Across {sleep_stage_label}', fontsize=20, pad=30)  # Add padding
    else:
        plt.title('Overall Bout Durations Across DataFrames', fontsize=20, pad=30)

    # Dynamically set y-axis limits based on the distribution of filtered stripplot data
    plt.ylim(0, y_max)

    # Perform Tukey's HSD if multiple dataframes are available
    if len(labels) > 1:
        tukey = tukey_test(bout_durations_dict, sleep_stage_label)
        comparisons = tukey._results_table.data[1:]  # Extract results from Tukey's test
        significance_threshold = 0.05  # Significance level for stars

        # Annotate significance stars on pairwise comparisons
        for comparison in comparisons:
            group1, group2, meandiff, p_adj, lower, upper, reject = comparison
            if p_adj < significance_threshold:  # Only annotate significant differences
                idx1 = labels.index(group1)
                idx2 = labels.index(group2)
                label_position = (idx1 + idx2) / 2  # Position between the two bars

                y_position = y_max - 0.1 * y_max  # Adjusted bracket position
                star_position = y_position + 0.05 * y_position  # Asterisk position slightly above the bracket

                plt.text(label_position, star_position, '*', ha='center', va='bottom', fontsize=12)

                # Add brackets closer to the asterisk
                plt.plot([idx1, idx1, idx2, idx2], 
                         [y_position, y_position + 0.05 * y_position, y_position + 0.05 * y_position, y_position], 
                         lw=1.5, color='black')

    f_stat, p_value = perform_anova(bout_durations_dict, sleep_stage_label)

    # Remove top and right spines for cleaner plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust spacing to fit the title and plot comfortably
    plt.tight_layout()  # Adjust to prevent overlap with title

    plt.show()

def plot_bout_duration_boxplot_all_data(bout_durations_dict, sleep_stage_label):
    plt.figure(figsize=(14, 8))
    
    labels = list(bout_durations_dict.keys())
    data_to_plot = []

    # Collect data for boxplot
    for df_name in labels:
        data = bout_durations_dict[df_name]
        if sleep_stage_label == 'All Stages':
            data = data['BoutDuration'].tolist()
        data_to_plot.append(data)

    # Define custom colors for boxplot
    box_colors = ['lightblue', 'orange', 'lightgreen', 'red']

    # Create boxplot
    sns.boxplot(data=data_to_plot, notch=False, whis=1.5, palette=box_colors[:len(labels)], showfliers=False, width=0.5)

    # Format x-axis labels
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, fontsize=14)
    plt.ylabel('Bout Duration (seconds)', fontsize=14)

    # Add title
    if sleep_stage_label:
        plt.title(f'Somnotate Performance - Bout Durations Across {sleep_stage_label}', fontsize=18, pad=30)
    else:
        plt.title('Overall Bout Durations Across DataFrames', fontsize=18, pad=30)

    # Remove top and right spines for cleaner plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 0))  # Set y-axis at the bottom of the graph

    # Adjust spacing to fit the title and plot comfortably
    plt.tight_layout()
    
    plt.show()


def count_transitions(df, df_name):
    ''' 
    Calculate the number of transitions between sleep stages in a CSV file.
    Input:
        df: DataFrame for the CSV file
    Output
        n_transitions_all: Dictionary with the number of transitions for each CSV file
    '''

    n_transitions_all = {}
    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]
    n_transitions = len(stage_changes)
    n_transitions_all[df_name] = n_transitions
    print(f'The number of transitions for {df_name} is {n_transitions}')

    return n_transitions_all

def count_REM_to_non_REM_transitions(df, df_name):
    ''' 
    Calculate the number of transitions from non-REM to REM sleep stages in a CSV file.
    Input:
        df: DataFrame for the CSV file
    Output
        n_incorrect_transitions_all : Dictionary with the number of non-REM to REM transitions for each CSV file                  
    '''

    n_incorrect_transitions_all = {}
    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]
    n_incorrect_transitions = 0
    for i in range(len(stage_changes)-1):
        if df['sleepStage'].iloc[stage_changes[i]] == 3 and df['sleepStage'].iloc[stage_changes[i+1]] == 2:
            n_incorrect_transitions += 1

    n_incorrect_transitions_all[df_name] = n_incorrect_transitions
    print(f'The number of non-REM to REM transitions for {df_name} is {n_incorrect_transitions}')



    return n_incorrect_transitions_all

def count_REM_to_awake_transitions(df, df_name):
    ''' 
    Calculate the number of transitions from REM to awake sleep stages in a CSV file.
    Input:
        df: DataFrame for the CSV file
    Output
        n_REM_to_awake_transitions_all : Dictionary with the number of REM to awake transitions for each CSV file                  
    '''

    n_REM_to_awake_transitions_all = {}
    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]
    n_REM_to_awake_transitions = 0
    for i in range(len(stage_changes)-1):
        if df['sleepStage'].iloc[stage_changes[i]] == 3 and df['sleepStage'].iloc[stage_changes[i+1]] == 1:
            n_REM_to_awake_transitions += 1

    n_REM_to_awake_transitions_all[df_name] = n_REM_to_awake_transitions
    print(f'The number of REM to awake transitions for {df_name} is {n_REM_to_awake_transitions}')

    return n_REM_to_awake_transitions_all

def count_non_REM_to_awake_transitions(df, df_name):
    '''
    Calculate the number of transitions from non-REM to awake sleep stages in a CSV file.
    Input:
        df: DataFrame for the CSV file
    Output:
        n_non_REM_to_awake_transitions_all : Dictionary with the number of non-REM to awake transitions for each CSV file
    '''

    n_non_REM_to_awake_transitions_all = {}
    stage_changes = np.where(df['sleepStage'].diff() != 0)[0]
    n_non_REM_to_awake_transitions = 0
    for i in range(len(stage_changes)-1):
        if df['sleepStage'].iloc[stage_changes[i]] == 2 and df['sleepStage'].iloc[stage_changes[i+1]] == 1:
            n_non_REM_to_awake_transitions += 1

    n_non_REM_to_awake_transitions_all[df_name] = n_non_REM_to_awake_transitions    
    print(f'The number of non-REM to awake transitions for {df_name} is {n_non_REM_to_awake_transitions}')

    return n_non_REM_to_awake_transitions_all


def plot_transitions(n_transitions_all):
    ''' 
    Plot the number of transitions between sleep stages for each CSV file.
    Input:
        n_transitions_all: Dictionary with the number of transitions for each CSV file
    '''

    df_names = list(n_transitions_all.keys())
    print(df_names)
    n_transitions_values = [list(item.values())[0] for item in n_transitions_all.values()]
    print(n_transitions_values)

    plt.bar(df_names, n_transitions_values, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
    plt.ylim(0, 180)
    plt.ylabel('Number of transitions')
    plt.show()





















































