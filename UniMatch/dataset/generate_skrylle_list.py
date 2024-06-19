import random


def split_files(input_file, output_file1, output_file2, num_lines_file1):
    # Read lines from the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Shuffle the lines randomly
    random.shuffle(lines)

    # Select lines for the first file
    lines_file1 = lines[:num_lines_file1]

    # Remaining lines for the second file
    lines_file2 = lines[num_lines_file1:]

    # Write lines to the output files
    with open(output_file1, 'w') as f:
        f.writelines(lines_file1)

    with open(output_file2, 'w') as f:
        f.writelines(lines_file2)


# Example usage:
input_file = 'unlabeled.txt'  # Input file containing directories
output_file1 = 'skrylle_test.txt'  # First output file
output_file2 = 'labeled.txt'  # Second output file
num_lines_file1 = 100  # Number of lines to put in the first file

split_files(input_file, output_file1, output_file2, num_lines_file1)
