#!/work/scratch/as41vomu/mambaforge/envs/pyiron/bin/python
import re

# Define the input file path
input_file = "./scf.out"


# check electronic convergence for each SCF cycle


# Function to read and process the file
def extract_scf_accuracy(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            counter = 0
            last_accuracy = None
            for line in lines:
                # Check for "End of self-consistent calculation"
                if "End of self-consistent calculation" in line:
                    if last_accuracy:
                        counter += 1
                        print(f"{counter} {last_accuracy}")

                # Check for "estimated scf accuracy" and update the last_accuracy variable
                match = re.search(
                    r"estimated scf accuracy\s*<\s*([\d\.E\-]+)\s*Ry", line
                )
                if match:
                    last_accuracy = float(match.group(1))

    except FileNotFoundError:
        print(f"File not found: {file_path}")


# Call the function
extract_scf_accuracy(input_file)
