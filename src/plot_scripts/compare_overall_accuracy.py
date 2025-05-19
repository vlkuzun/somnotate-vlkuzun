import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import functions_for_somno_QM_checks as QMfunctions


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import functions_for_somno_QM_checks as QMfunctions


def main(paths, save_path=None, dpi=600):
    '''
    Compare the overall similarity of sleep stage annotations between different files.
    Input:
    - paths: dictionary, where the keys are the paths to the CSV files and the values are the labels for the files.
    - save_path: optional str, path to save the figure. If None, the figure is displayed instead.
    - dpi: int, resolution for saved figure (default 600)
    Output:
    - Displays or saves a heatmap showing the similarity matrix of sleep stage annotations.
    '''
    # Increase default font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 22,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    
    dataframes = {QMfunctions.rename_file(path): pd.read_csv(path) for path in paths}

    labels = list(dataframes.keys())  # Extract labels from the dictionary
    matrix = np.zeros((len(labels), len(labels)))  # Initialize a matrix to store the similarity scores

    # Compare each pair of files and store the similarity score in the matrix
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            df1, df2 = QMfunctions.match_length_csv_files(dataframes[label1], dataframes[label2])
            similarity = QMfunctions.compare_csv_files(df1, df2)
            matrix[i, j] = similarity  

    # Create the heatmap
    plt.figure(figsize=(10, 8))  # Slightly larger figure
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, 
                cbar=True, annot_kws={"size": 18})  # Increased annotation size
    plt.tick_params(axis='x', labelsize=18, rotation=45)  # Increased tick label size and rotated for better readability
    plt.tick_params(axis='y', labelsize=18)

    # Customize the color bar font size
    colorbar = plt.gca().collections[0].colorbar
    colorbar.ax.tick_params(labelsize=16)  # Increased colorbar tick size
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path} with DPI={dpi}")
    else:
        plt.show()


if __name__ == "__main__":
    
    paths_input = input("Enter the paths to the csv files with annotations (without quotes) and their labels (format: path,label;path,label): ")
    
    paths = {}
    for entry in paths_input.split(";"):
        path, label = entry.split(",")
        paths[path.strip()] = label.strip()
    
    save_option = input("Do you want to save the figure? (yes/no): ").strip().lower()
    if save_option == 'yes':
        save_dir = input("Enter directory to save the figure (or press Enter for current directory): ").strip()
        save_filename = input("Enter filename (without extension): ").strip()
        dpi_input = input("Enter DPI (default is 600): ").strip()
        
        if not save_dir:
            save_dir = os.getcwd()
        
        dpi = 600 if not dpi_input else int(dpi_input)
        save_path = os.path.join(save_dir, f"{save_filename}.png")
        main(paths, save_path, dpi)
    else:
        main(paths)



# Example input:
# paths_input = Y:/Francesca/somnotate/checking_accuracy/somno_csv/automated_state_annotationoutput_sub-010_ses-01_recording-01.csv,somnotate;Y:/Francesca/somnotate/train_set/train_csv_files/sub-003_ses-02_recording-01.csv,control;Y:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_fp.csv,fp
# paths_input_v2  = /Volumes/harris/Francesca/somnotate/train_set/train_csv_files/sub-003_ses-02_recording-01.csv,control;/Volumes/harris/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_fp.csv,fp;/Volumes/harris/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_export(HBH).csv,bh;/Volumes/harris/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_vu.csv,vu;/Volumes/harris/Francesca/somnotate/checking_accuracy/somno_csv/automated_state_annotationoutput_sub-010_ses-01_recording-01.csv,somnotate

# Example paths:
# path1 = "Y:/Francesca/somnotate/checking_accuracy/somno_csv/automated_state_annotationoutput_sub-010_ses-01_recording-01.csv"
# path2 = "Y:/Francesca/somnotate/train_set/train_csv_files/sub-003_ses-02_recording-01.csv" # unmatched file used as a control 
# path3 = "Y:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_fp.csv"
# path4 = "Y:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_export(HBH).csv"
# path5 = "Y:/Francesca/somnotate/checking_accuracy/manual_csv/sub-010_ses-01_recording-01_data-sleepscore_vu.csv"
