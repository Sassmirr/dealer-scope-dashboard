import pandas as pd
import os

# Define file paths relative to the current working directory (e.g., dealer/)
input_file1 = '../data/raw/FS-data-80475.csv'
input_file2 = '../data/raw/FS-data-80475-2025-all-months.csv'
output_folder = '../data/processed/'
output_file = 'cleaned_dataset.csv'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Read the datasets
try:
    df1 = pd.read_csv(input_file1)
    df2 = pd.read_csv(input_file2)
    
    # Merge the two datasets
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # Data Cleaning
    merged_df.drop_duplicates(inplace=True)
    merged_df.dropna(subset=['english_name'], inplace=True)
    
    # Convert value columns to numeric
    merged_df['monthly_value'] = pd.to_numeric(merged_df['monthly_value'], errors='coerce')
    merged_df['yearly_value'] = pd.to_numeric(merged_df['yearly_value'], errors='coerce')
    merged_df.dropna(subset=['monthly_value', 'yearly_value'], inplace=True)
    
    # Create a proper datetime column
    merged_df['date'] = pd.to_datetime(merged_df['year'].astype(str) + '-' + merged_df['month'].astype(str) + '-01')
    
    # Sort the data by date and account_id
    merged_df.sort_values(by=['account_id', 'date'], inplace=True)
    
    # Save the cleaned dataset to the specified folder
    final_path = os.path.join(output_folder, output_file)
    merged_df.to_csv(final_path, index=False)
    
    print(f"Successfully created and saved the cleaned dataset to '{final_path}'.")
    print(f"Final number of rows: {len(merged_df)}")
    
except FileNotFoundError:
    print("One or more input files not found. Please ensure 'FS-data-80475.csv' and 'FS-data-80475-2025-all-months.csv' are in the same directory as this script.")
except Exception as e:
    print(f"An error occurred: {e}")