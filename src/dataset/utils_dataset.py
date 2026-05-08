import os
import re

# ─── Defines function to find the next possible name
#
def get_next_filename(output_folder, base_name, type):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_index = 0
    
    while True:
        output_filename = f"{base_name}_{image_index}.{type}"
        output_path = os.path.join(output_folder, output_filename)
        
        if not os.path.exists(output_path):
            return output_filename
            
        image_index += 1

# ─── Function to extract the numeric part from the filename
#
def extract_number(filename):   
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else 0

# ─── Function to sort file paths based on the numeric part
#
def sort_files_numerically(file_paths):
    return sorted(file_paths, key=lambda x: extract_number(x))

# ─── Function to get the next available test directory
#
def get_next_test_dir(base_dir="output"):
    n = 1
    while True:
        test_dir = os.path.join(base_dir, f"test{n}")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            return test_dir
        n += 1
