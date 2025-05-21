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

    # Plot the pie chart
    plt.figure(figsize=(8, 6))  # Increased figure size
    plt.rcParams.update({'font.size': 20})  # Set base font size for all elements
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 22},  # Increased label font size
            colors=['#ff9999','#66b3ff','#99ff99'], wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    # Increase percentage value font size
    for text in plt.gca().texts:
        text.set_fontsize(20)  # Increased percentage font size
        
    plt.title(title, fontsize=26, pad=20)  # Increased title font size and added padding

    # Generate output path from the filename with EPS extension
    output_path = os.path.join(output_dir, f"{filename}_sleep_stage_pie_chart.eps")
    
    # Save the pie chart as an EPS file (vector format)
    plt.savefig(output_path, format='eps', bbox_inches='tight')
    print(f"Pie chart saved to: {output_path}")
    
    # Display the plot
    plt.show()
    
    # Close the plot after displaying
    plt.close()

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