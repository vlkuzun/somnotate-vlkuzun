import csv

def convert_paths_in_csv(file_path, current_value, new_value):
    """
    Opens a CSV file and converts any cell containing the specified current_value to the new_value.

    Args:
        file_path (str): The path to the CSV file to be processed.
        current_value (str): The value to search for in each cell.
        new_value (str): The value to replace the current_value with.

    Returns:
        None
    """
    # Read the CSV content
    with open(file_path, 'r') as csv_file:
        reader = list(csv.reader(csv_file))

    # Modify the cells
    modified_rows = []
    for row in reader:
        modified_row = [cell.replace(current_value, new_value, 1) if cell.startswith(current_value) else cell for cell in row]
        modified_rows.append(modified_row)

    # Write the modified content back to the CSV
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(modified_rows)

# Example usage
convert_paths_in_csv("/Volumes/harris/somnotate/to_score_set/to_score_sheet.csv", '/Volumes/harris-1', '/Volumes/harris')
