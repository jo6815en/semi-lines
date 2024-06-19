import random

# Open the input file in read mode
#with open('wireframe_train.txt', 'r') as input_file:
    # Open the output file in write mode
#    with open('output.txt', 'w') as output_file:
        # Read each line from the input file
#        for line in input_file:
            # Write the line to the output file along with additional text
#            output_file.write("./data/wireframe_raw/images/" + line.strip() + "\n")

import random

# Specify the number of lines for validation
validation_lines_count = 300
# Specify the maximum number of lines for output.txt
max_output_lines = 74

# Open the input file in read mode
with open('wireframe_train.txt', 'r') as input_file:
    # Read all lines from the input file
    lines = input_file.readlines()

    # Take the first 300 lines for validation
    validate_lines = lines[:validation_lines_count]
    # Take the rest of the lines for training
    remaining_lines = lines[validation_lines_count:]

    # Randomly shuffle the remaining lines for training
    random.shuffle(remaining_lines)

    # Open the output file in write mode for validation data
    with open('val.txt', 'w') as validate_file:
        # Write the first 300 lines to the validation file
        for line in validate_lines:
            validate_file.write("./data/wireframe_raw/images/" + line.strip() + "\n")

    # Open the output file in write mode for training data
    with open('labeled.txt', 'w') as output_file:
        # Write a maximum of max_output_lines to the output file
        for line in remaining_lines[:max_output_lines]:
            output_file.write("./data/wireframe_raw/images/" + line.strip() + "\n")

    # Open the output file in write mode for unlabeled data
    with open('unlabeled.txt', 'w') as unlabeled_file:
        # Write the remaining lines to the unlabeled file
        for line in remaining_lines[max_output_lines:]:
            unlabeled_file.write("./data/wireframe_raw/images/" + line.strip() + "\n")
