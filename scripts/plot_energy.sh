#!/bin/bash
ml gnuplot
# Check if OSZICAR file exists
if [ ! -f OSZICAR ]; then
  echo "OSZICAR file not found in the current directory."
  exit 1
fi

# Check if gnuplot is installed
if ! command -v gnuplot >/dev/null; then
  echo "gnuplot is not installed. Please install gnuplot and try again."
  exit 1
fi

# Extract energy and ionic step data from OSZICAR
awk '/F=/ {print NR, $5}' OSZICAR > energy_vs_N.dat

# Define colors for each ionic step (extend the list as needed)
colors="blue,red,green,cyan,magenta,yellow,black"

# Create a temporary gnuplot script
# Create a temporary gnuplot script
cat > gnuplot_script.gp << EOL
set terminal pngcairo size 800,600 enhanced font 'Verdana,10'
set output 'energy_vs_N.png'
set title 'Energy vs N for each Ionic Step'
set xlabel 'N'
set ylabel 'Energy (E)'
unset key
set grid
set format x "%.0f"
set format y "%.6e"
set mxtics 2
set mytics 2
set style line 1 lt 1 lw 1 lc rgb "blue"
set style line 2 lt 1 lw 1 lc rgb "red"
set style line 3 lt 1 lw 1 lc rgb "green"
set style line 4 lt 1 lw 1 lc rgb "cyan"
set style line 5 lt 1 lw 1 lc rgb "magenta"
set style line 6 lt 1 lw 1 lc rgb "yellow"
set style line 7 lt 1 lw 1 lc rgb "black"
color_array = "blue,red,green,cyan,magenta,yellow,black"
array_len = words(color_array)
stats 'energy_vs_N.dat' using 1 nooutput
n=STATS_records
plot for [i=1:n] 'energy_vs_N.dat' every ::0::i-1 with linespoints pt 7 ps 1 lc (i % array_len + 1)

EOL

# Run the gnuplot script
gnuplot gnuplot_script.gp

# Clean up temporary files
rm gnuplot_script.gp
rm energy_vs_N.dat

echo "Plot saved as energy_vs_N.png in the current directory."
