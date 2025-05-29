import os
import pandas as pd
import matplotlib.pyplot as plt

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

def create_pie_chart(df, output_dir, filename, title):
    # Count the occurrences of each sleep stage
    stage_counts = df['sleepStage'].value_counts()

    # Define the labels and sizes for the pie chart
    labels = ['Wake', 'NREM', 'REM']
    sizes = [stage_counts.get(1, 0), stage_counts.get(2, 0), stage_counts.get(3, 0)]
    
    # Calculate the exact percentages
    total = sum(sizes)
    exact_percentages = [(s/total)*100 for s in sizes]
    
    # Round percentages to whole numbers, ensuring they sum to 100%
    percentages = [round(p) for p in exact_percentages]
    
    # Adjust the largest value to ensure sum is 100%
    diff = 100 - sum(percentages)
    if diff != 0:
        max_idx = exact_percentages.index(max(exact_percentages))
        percentages[max_idx] += diff
    
    # Create percentage labels for each segment
    autopct_labels = [f"{p}%" for p in percentages]
    
    # Plot the pie chart - use rcParams for figure size
    plt.figure()
    
    # Define consistent colors for sleep stages
    colors = {
        'Wake': '#FFFDD0',  # Cream color for Wake
        'NREM': '#ADD8E6',  # Light blue for NREM
        'REM': '#90EE90'    # Light green for REM
    }
    
    # Use manual labels with font sizes from rcParams
    plt.pie(sizes, labels=labels, autopct=lambda pct, allvals=sizes, idx=[0,1,2]: autopct_labels[idx.pop(0)],
            startangle=140,
            colors=[colors['Wake'], colors['NREM'], colors['REM']], 
            wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    
    plt.title(title)

    # Generate output paths for PNG and PDF formats
    base_filename = f"{filename}_sleep_stage_pie_chart"
    png_output_path = os.path.join(output_dir, f"{base_filename}.png")
    pdf_output_path = os.path.join(output_dir, f"{base_filename}.pdf")
    
    # Save the pie chart using rcParams DPI settings
    plt.savefig(png_output_path, bbox_inches='tight')
    plt.savefig(pdf_output_path, format='pdf', bbox_inches='tight')
    
    plt.close()
    print(f"Pie chart saved as PNG: {png_output_path}")
    print(f"Pie chart saved as PDF: {pdf_output_path}")

def main():
    print("Welcome to the Sleep Stage Pie Chart Generator!")

    # Get CSV file path from the user
    while True:
        csv_file = input("Enter the path of a CSV file (or type 'done' to finish): ")
        if csv_file.lower() == 'done':
            break
        elif os.path.exists(csv_file):
            # Get the output directory path
            output_dir = input("Enter the output directory for the pie charts: ")
            if not os.path.exists(output_dir):
                print(f"Error: The directory '{output_dir}' does not exist. Please try again.")
                continue

            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)

                # Check if 'sleepStage' column exists
                if 'sleepStage' not in df.columns:
                    print(f"Error: 'sleepStage' column not found in {csv_file}. Skipping this file.")
                    continue

                # Ask user for subject, session, recording, and extra info
                subject = input("Enter subject: ")
                session = input("Enter session: ")
                recording = input("Enter recording: ")
                extra_info = input("Enter extra information: ")

                # Ask user to use subject and session for title or custom title
                use_default_title = input("Would you like to use the subject and session as the title? (yes/no): ").strip().lower()
                if use_default_title == "yes":
                    title = f'Sleep stage distribution for {subject}_{session} across entire recording'
                else:
                    title = input("Enter a custom title for the chart: ")

                # Construct a filename for the pie chart
                filename = f"{subject}_{session}_{recording}_{extra_info}"

                # Create pie chart for the current file
                create_pie_chart(df, output_dir, filename, title)

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

        else:
            print(f"Error: The file at '{csv_file}' does not exist. Please try again.")

if __name__ == "__main__":
    main()