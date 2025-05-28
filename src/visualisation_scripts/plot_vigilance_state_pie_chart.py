import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the Seaborn theme
#sns.set_theme(style="whitegrid")

def create_pie_chart(df, output_dir, filename, title):
    # Count the occurrences of each sleep stage
    stage_counts = df['sleepStage'].value_counts()

    # Define the labels and sizes for the pie chart
    labels = ['Wake', 'NREM', 'REM']  # Changed 'Awake' to 'Wake'
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
    
    # Plot the pie chart
    plt.figure(figsize=(8, 6))  # Increased figure size
    plt.rcParams.update({'font.size': 20})  # Set base font size for all elements
    
    # Use manual labels instead of a custom autopct function
    plt.pie(sizes, labels=labels, autopct=lambda pct, allvals=sizes, idx=[0,1,2]: autopct_labels[idx.pop(0)],
            startangle=140, textprops={'fontsize': 22},
            colors=['#ff9999','#66b3ff','#99ff99'], wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    # Increase percentage value font size
    for text in plt.gca().texts:
        text.set_fontsize(20)  # Increased percentage font size
        
    plt.title(title, fontsize=26, pad=20)  # Increased title font size and added padding

    # Generate output paths for PNG and EPS formats
    base_filename = f"{filename}_sleep_stage_pie_chart"
    png_output_path = os.path.join(output_dir, f"{base_filename}.png")
    eps_output_path = os.path.join(output_dir, f"{base_filename}.eps")
    
    # Save the pie chart as PNG with high DPI
    plt.savefig(png_output_path, dpi=600, bbox_inches='tight')
    
    # Save the pie chart as EPS (vector format for publication)
    plt.savefig(eps_output_path, format='eps', bbox_inches='tight')
    
    plt.close()
    print(f"Pie chart saved as PNG: {png_output_path}")
    print(f"Pie chart saved as EPS: {eps_output_path}")

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