#This module is for keeping the data sections of gyroscope, accelerometer, magnetometer and dropping other sections. After the truncation, it converts the txt files to csv format.

import pandas as pd
import os

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def convert_txt_to_csv(file_paths, columns_to_keep, output_base_dir):
    for file_path in file_paths:
        try:
            print(f"Processing file: {file_path}")
            
            # Read the file, ignoring comment lines starting with '//'
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Filter out comment lines
            data_lines = [line for line in lines if not line.startswith('//')]
            
            # Save the filtered lines to a temporary file
            temp_file_path = file_path.replace('.txt', '_temp.txt')
            with open(temp_file_path, 'w') as temp_file:
                temp_file.writelines(data_lines)
            
            # Read the filtered data
            data = pd.read_csv(temp_file_path, sep='\t')
            
            # Select the required columns
            truncated_data = data[columns_to_keep]
            
            # Generate the relative path for the output file
            relative_path = os.path.relpath(file_path, input_dir)
            output_file_path = os.path.join(output_base_dir, relative_path.replace('.txt', '.csv'))
            
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_file_path)
            ensure_directory_exists(output_dir)
            
            # Save the data as a CSV file
            truncated_data.to_csv(output_file_path, index=False)
            print(f"Saved CSV to: {output_file_path}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
        finally:
            # Remove the temporary file if it exists
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Removed temporary file: {temp_file_path}")
    
    print("Conversion to CSV completed.")

def get_all_txt_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_paths.append(os.path.join(root, file))
    return file_paths

# Define the base input directory
input_dir = r'C:\Users\reyna\Desktop\BSN Project Dataset\Tests\104\104\Testler Export'

# Get all .txt files in the input directory and its subdirectories
file_paths = get_all_txt_files(input_dir)

# Check if any files are found
if not file_paths:
    print("No .txt files found in the directory.")

# Print the list of files found for debugging
print(f"Found {len(file_paths)} .txt files.")

# Columns to keep
columns_to_keep = ['Counter', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']

# Define the base output directory
output_base_dir = r'C:\Users\reyna\Desktop\BSN Project Dataset\Preprocessed\newPredict'

# Ensure the base output directory exists
ensure_directory_exists(output_base_dir)

# Convert the files
convert_txt_to_csv(file_paths, columns_to_keep, output_base_dir)
