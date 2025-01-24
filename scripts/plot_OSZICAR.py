import os
import argparse
import re
import matplotlib.pyplot as plt

plt.style.use('/home/as41vomu/.config/matplotlib/stylelib/highres_science.mplstyle')

parser = argparse.ArgumentParser(description="Plot energies from OSZICAR file")
parser.add_argument("--skip", type=int, default=0, help="Number of steps to skip in the first list")
args = parser.parse_args()

filename = "OSZICAR"
skip_steps = args.skip

energy_data = []
ionic_step_data = []

energy_pattern = re.compile(r"^[A-Z]+:\s+\d+\s+(-?\d+\.\d+E[+-]\d+)\s+")

with open(filename, "r") as file:
    for line in file:
        match = energy_pattern.match(line)
        if match:
            e = float(match.group(1))
            energy_data.append(e)
        elif "1 F=" in line or "E0=" in line:
            ionic_step_data.append(energy_data)
            energy_data = []

ionic_step_data[0] = ionic_step_data[0][skip_steps:]

cmap = plt.get_cmap("tab10")
colors = cmap(range(len(ionic_step_data)))

cumulative_electronic_steps = 0

for idx, energies in enumerate(ionic_step_data):
    x_values = [step + cumulative_electronic_steps for step in range(len(energies))]
    plt.plot(x_values, energies, marker="o", linestyle="--", color=colors[idx], label=f"Ionic step {idx + 1}")
    cumulative_electronic_steps += len(energies)

plt.title(f"Energy plot for {os.path.basename(os.getcwd())}")
plt.xlabel("Cumulative electronic steps")
plt.ylabel("Energy")
plt.legend()

plot_filename = "energy_plot.pdf"
plt.savefig(plot_filename)
plt.tight_layout()
plt.show()

