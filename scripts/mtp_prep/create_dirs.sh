#!/bin/bash

# Define the root directory
root_dir="test"

# Define the subdirectories
sub_dirs=("fit_0" "fit_1")
fit_sub_dirs=("active_learning" "dbs" "dft_calculations" "mtp_fit")
dbs_sub_dirs=("initial_db" "active_learning_db")
dft_sub_dirs=("static_initial" "active_learning")


# Create the root directory
mkdir $root_dir

# Copy the utils directory
cp -r "/home/as41vomu/scripts/mtp_prep/utils" "$root_dir/"

# Create the subdirectories
for sub_dir in "${sub_dirs[@]}"; do
  mkdir "$root_dir/$sub_dir"
  for fit_sub_dir in "${fit_sub_dirs[@]}"; do
    mkdir "$root_dir/$sub_dir/$fit_sub_dir"
    if [ "$fit_sub_dir" = "dbs" ]; then
      for dbs_sub_dir in "${dbs_sub_dirs[@]}"; do
        mkdir "$root_dir/$sub_dir/$fit_sub_dir/$dbs_sub_dir"
      done
    elif [ "$fit_sub_dir" = "dft_calculations" ]; then
      for dft_sub_dir in "${dft_sub_dirs[@]}"; do
        mkdir "$root_dir/$sub_dir/$fit_sub_dir/$dft_sub_dir"
      done
    fi
  done
done

# Create README.md and requirements.txt
touch "$root_dir/README.md"
touch "$root_dir/requirements.txt"

echo "Directory structure created."


# Copy the scripts
for sub_dir in "${sub_dirs[@]}"; do
  for fit_sub_dir in "${fit_sub_dirs[@]}"; do
    if [ "$fit_sub_dir" = "mtp_fit" ]; then
      cp "$root_dir/utils/mtp_fit_scripts/"* "$root_dir/$sub_dir/$fit_sub_dir/"
    elif [ "$fit_sub_dir" = "dft_calculations" ]; then
      for dft_sub_dir in "${dft_sub_dirs[@]}"; do
        cp "$root_dir/utils/dft_calculations_scripts/"* "$root_dir/$sub_dir/$fit_sub_dir/$dft_sub_dir/"
      done
    elif [ "$fit_sub_dir" = "active_learning" ]; then
      cp "$root_dir/utils/active_learning_scripts/"* "$root_dir/$sub_dir/$fit_sub_dir/"
    elif [ "$fit_sub_dir" = "dbs" ]; then
      for dbs_sub_dir in "${dbs_sub_dirs[@]}"; do
        cp "$root_dir/utils/dbs_conversion_scripts/"* "$root_dir/$sub_dir/$fit_sub_dir/$dbs_sub_dir/"
      done
    fi
  done
done

echo "Scripts copied."

