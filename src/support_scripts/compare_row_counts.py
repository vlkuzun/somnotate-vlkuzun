import csv

def count_rows_in_csv(file1, file2):
    # Count rows in first CSV file
    def count_rows(file):
        with open(file, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for row in reader)  # Counting rows
        return row_count
    
    # Count rows in both files
    count_file1 = count_rows(file1)
    count_file2 = count_rows(file2)

    return count_file1, count_file2

# Example usage:
file1 = '/ceph/harris/somnotate/to_score_set/to_score_csv_files/sub-007_ses-01_recording-01_time-0-70.5h.csv'
file2 = '/ceph/harris/somnotate/to_score_set/vis_back_to_csv/automated_state_annotationoutput_sub-007_ses-01_recording-01_time-0-70.5h_512Hz.csv'
rows_file1, rows_file2 = count_rows_in_csv(file1, file2)
print(f'Rows in {file1}: {rows_file1}')
print(f'Rows in {file2}: {rows_file2}')
