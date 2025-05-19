import pandas as pd
import functions_for_somno_QM_checks as QMfunctions  


def main():
    # Collect paths from user input
    paths_input = input("Enter the paths to the CSV files without quotes, separated by commas: ")
    paths = [path.strip() for path in paths_input.split(",")]  

    # Example paths (you can change these accordingly)
    # path1 = "Z:/Francesca/somnotate/checking_accuracy/somno_csv/automated_state_annotationoutput_sub-010_ses-01_recording-01.csv"
    # path2 = "Z:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_fp.csv"
    # path3 = "Z:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_export(HBH).csv"
    # path4 = "Z:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_vu.csv"
    # combined path = Z:/Francesca/somnotate/checking_accuracy/somno_csv/automated_state_annotationoutput_sub-010_ses-01_recording-01.csv,Z:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_fp.csv,Z:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_export(HBH).csv,Z:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_vu.csv
    # combined path mac = /Volumes/harris/Francesca/somnotate/checking_accuracy/somno_csv/automated_state_annotationoutput_sub-010_ses-01_recording-01.csv,/Volumes/harris/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_fp.csv,/Volumes/harris/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_export(HBH).csv,/Volumes/harris/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_vu.csv

    bout_durations_with_stage_all = {}

    for path in paths:
        df_name = QMfunctions.rename_file(path)  # Generate the label for the file based on its name 
        try:
            # Load the CSV file into a DataFrame
            df = pd.read_csv(path, usecols=['sleepStage'])  # Load only the 'sleepStage' column
            print(f"Loaded {df_name} successfully from {path}.")  # Confirm that the file is loaded
            
            # Check if 'sleepStage' column exists in the DataFrame
            if 'sleepStage' not in df.columns:
                raise ValueError(f"The 'sleepStage' column is missing in {df_name}.")
            
            bout_durations = QMfunctions.get_bout_durations(df, sampling_rate=512, df_name=df_name)
            # Store the bout durations in the main dictionary using df_name as key
            bout_durations_with_stage_all[df_name] = bout_durations[df_name]  # Use df_name as the key
            print(f'Successfully calculated bout durations for {df_name}.')
            
        except Exception as e:
            print(f"Error loading {df_name} from {path}: {e}")  # Print any error encountered

    # If we have successfully loaded the dataframes, compare bout durations
    if bout_durations_with_stage_all:
        bout_durations_awake, bout_durations_nrem, bout_durations_rem = QMfunctions.get_stage_durations(bout_durations_with_stage_all)
        print('Successfully calculated bout durations by stage.')

        # Run ANOVA tests for each stage
        #QMfunctions.perform_anova(bout_durations_with_stage_all, 'All Stages')
        #QMfunctions.perform_anova(bout_durations_awake, sleep_stage_label='Awake')
        #QMfunctions.perform_anova(bout_durations_nrem, sleep_stage_label='NREM')
        #QMfunctions.perform_anova(bout_durations_rem, sleep_stage_label='REM')
        #print('Successfully ran ANOVA tests.')

        # Run Tukey tests for each stage
        #QMfunctions.tukey_test(bout_durations_with_stage_all, 'All Stages')
        #QMfunctions.tukey_test(bout_durations_awake, 'Awake')
        #QMfunctions.tukey_test(bout_durations_nrem, 'NREM')
        #QMfunctions.tukey_test(bout_durations_rem, 'REM')
        #print('Successfully ran Tukey test.')

        # Plot histograms for each stage with significance
        QMfunctions.plot_bout_duration_histograms_with_significance(bout_durations_with_stage_all, 'All Stages')
        QMfunctions.plot_bout_duration_histograms_with_significance(bout_durations_awake, 'Awake')
        QMfunctions.plot_bout_duration_histograms_with_significance(bout_durations_nrem, 'NREM')
        QMfunctions.plot_bout_duration_histograms_with_significance(bout_durations_rem, 'REM')
        print('Successfully plotted barplot with significance.')


# Run the main function
if __name__ == "__main__":
    main()




