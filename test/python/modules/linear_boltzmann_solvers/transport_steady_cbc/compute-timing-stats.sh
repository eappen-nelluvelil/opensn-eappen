# #!/bin/bash

# # This script runs a program multiple times, extracts specific timing data
# # from its output, and calculates the average of that data.

# # Number of times to run the program
# N=10

# # Create temporary files to store the output values.
# # These files will be automatically deleted when the script exits.
# SWEEP_TIMES=$(mktemp)
# SWEEP_TIMES_PER_UNKNOWN=$(mktemp)
# trap 'rm -f "$SWEEP_TIMES" "$SWEEP_TIMES_PER_UNKNOWN"' EXIT

# # The command to be executed.
# # Use $HOME instead of ~ for the home directory.
# COMMAND="mpirun --np 4 $HOME/opensn-eappen/build-cbc-fluds-only-storing-angular-fluxes/python/opensn -i transport_2d_1_poly.py"

# echo "Running the command $N times..."

# # Loop N times to run the command and collect data.
# for i in $(seq 1 $N)
# do
#   echo "Running iteration $i..."
#   # Execute the command and save its output to a variable.
#   # The '2>&1' part ensures that both standard output and standard error are captured.
#   output=$($COMMAND 2>&1)

#   # Use awk to find the lines containing the desired text and print the last column
#   # of those lines to the temporary files.
#   echo "$output" | awk '/Average sweep time \(s\):/ {print $NF}' >> "$SWEEP_TIMES"
#   echo "$output" | awk '/Sweep Time\/Unknown \(ns\):/ {print $NF}' >> "$SWEEP_TIMES_PER_UNKNOWN"
# done

# echo "----------------------------------------"
# echo "Calculating Averages..."
# echo "----------------------------------------"

# # --- NEW: Error Checking ---
# # Check if the temporary files are empty before trying to calculate.
# if [ ! -s "$SWEEP_TIMES" ] || [ ! -s "$SWEEP_TIMES_PER_UNKNOWN" ]; then
#     echo "Error: Failed to extract data from the program output."
#     echo "Please check the following:"
#     echo "1. The command is correct and runs without errors: $COMMAND"
#     echo "2. The output of the command contains the lines 'Average sweep time (s):' and 'Sweep Time/Unknown (ns):'"
#     exit 1
# fi

# # Use awk to calculate the average of the numbers in each temporary file.
# awk '{ total += $1 } END { print "Average sweep time (s): " total/NR }' "$SWEEP_TIMES"
# awk '{ total += $1 } END { print "Average Sweep Time/Unknown (ns): " total/NR }' "$SWEEP_TIMES_PER_UNKNOWN"

#!/bin/bash

# This script runs a program multiple times and computes the average of
# specific timing statistics for each "groupset" found in the output.

# --- Configuration ---
# Number of times to run the program
N=10

# The command to be executed.
# Use $HOME instead of ~ for the home directory.
COMMAND="mpirun --np 2 $HOME/opensn-eappen-memory-pool-allocator/opensn-eappen/build/python/opensn -i hdpe_balance.py"

# --- Script Logic ---
echo "Target command: $COMMAND"
echo "Running the command $N times..."

# Create temporary files to store the categorized output values.
# Format will be: <groupset_id> <value>
SWEEP_TIMES_DATA=$(mktemp)
TIME_PER_UNKNOWN_DATA=$(mktemp)

# Ensure temporary files are deleted when the script exits
trap 'rm -f "$SWEEP_TIMES_DATA" "$TIME_PER_UNKNOWN_DATA"' EXIT

# Loop N times to run the command and collect data.
for i in $(seq 1 $N)
do
  echo "Running iteration $i of $N..."
  # Execute the command and pipe its output directly to awk for processing.
  $COMMAND 2>&1 | awk \
    -v sweep_file="$SWEEP_TIMES_DATA" \
    -v tpu_file="$TIME_PER_UNKNOWN_DATA" \
    '
    # When we find the line indicating a new groupset is being solved...
    /^\s*\[0\].*Solving groupset/ {
        # ...we save the groupset ID (the 5th field in that line).
        current_groupset = $5
    }
    # When we find the "Average sweep time" line...
    /Average sweep time \(s\):/ {
        # ...append the current groupset ID and the value to the sweep time file.
        print current_groupset, $NF >> sweep_file
    }
    # When we find the "Sweep Time/Unknown" line...
    /Sweep Time\/Unknown \(ns\):/ {
        # ...append the current groupset ID and the value to the time-per-unknown file.
        print current_groupset, $NF >> tpu_file
    }
    '
done

echo "----------------------------------------"
echo "Calculating Averages per Groupset..."
echo "----------------------------------------"

# --- Error Checking ---
if [ ! -s "$SWEEP_TIMES_DATA" ]; then
    echo "Error: Failed to extract any data from the program output." >&2
    echo "Please check that the command is correct and the output format has not changed." >&2
    exit 1
fi

# --- Calculation and Output ---
echo "Average sweep time (s):"
# This awk command calculates the average for each groupset ID (the first field).
# It stores sums and counts in arrays keyed by the groupset ID.
awk '
{
    sum[$1] += $2;
    count[$1]++;
}
END {
    # Sort the groupset IDs numerically for clean, ordered output
    PROCINFO["sorted_in"] = "@ind_num_asc";
    for (gs_id in sum) {
        average = sum[gs_id] / count[gs_id];
        printf "  Groupset %s: %.6f\n", gs_id, average;
    }
}' "$SWEEP_TIMES_DATA"

echo "" # Add a blank line for readability

echo "Average Sweep Time/Unknown (ns):"
awk '
{
    sum[$1] += $2;
    count[$1]++;
}
END {
    PROCINFO["sorted_in"] = "@ind_num_asc";
    for (gs_id in sum) {
        average = sum[gs_id] / count[gs_id];
        printf "  Groupset %s: %.6f\n", gs_id, average;
    }
}' "$TIME_PER_UNKNOWN_DATA"