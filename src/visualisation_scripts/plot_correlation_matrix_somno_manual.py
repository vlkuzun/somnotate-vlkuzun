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


def main(paths, save_path=None):
    '''
    Compare the overall similarity of sleep stage annotations between different files.
    Input:
    - paths: dictionary, where the keys are the paths to the CSV files and the values are the labels for the files.
    - save_path: optional str, path to save the figure. If None, the figure is displayed instead.
    Output:
    - Displays or saves a heatmap showing the similarity matrix of sleep stage annotations.
    '''
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
    plt.figure(figsize=(6, 5))  # Adjusted for publication style
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, 
                cbar=True, annot_kws={"size": 10})  # Adjusted annotation size
    plt.tick_params(axis='x', rotation=45)  # Adjusted tick label size and rotated for better readability
    plt.tick_params(axis='y')

    # Customize the color bar font size
    colorbar = plt.gca().collections[0].colorbar
    colorbar.ax.tick_params(labelsize=10)  # Adjusted colorbar tick size
    
    plt.tight_layout()
    
    if save_path:
        # Extract base path without extension
        base_save_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        
        # Save as PNG format
        png_path = f"{base_save_path}.png"
        plt.savefig(png_path, bbox_inches='tight')
        
        # Save as PDF format instead of EPS for better font handling
        pdf_path = f"{base_save_path}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        print(f"Figure saved as PDF: {pdf_path}")
        print(f"Figure saved as PNG: {png_path}")
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
        save_path = input("Enter complete path to save the figure (with filename, without extension): ").strip()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        main(paths, save_path)
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
