import numpy as np
import matplotlib.pyplot as plt

# paths = ["[0.3x0.01]_10_1E+06_9.81.txt",
#          "[0.3x0.01]_10_1E+06_15.txt",
#          "[0.3x0.01]_10_1E+06_30.txt"]

# names = ["9.81 N/kg", "15 N/kg", "30 N/kg"]

# paths = ["[0.3x0.01]_10_1E+06_9.81.txt",
#          "[0.3x0.01]_10_1E+07_9.81.txt",
#          "[0.3x0.01]_10_1E+08_9.81.txt"]

# names = ["1E+06 GPa", "1E+07 GPa", "1E+08 GPa"]

# paths = ["[0.3x0.01]_10_1E+06_9.81.txt",
#          "[0.3x0.01]_15_1E+06_9.81.txt",
#          "[0.3x0.01]_30_1E+06_9.81.txt"]

# names = ["10 kg/m^3", "15 kg/m^3", "30 kg/m^3"]

# paths = ["[0.3x0.01]_10_1E+06_9.81.txt",
#         "[0.4x0.01]_10_1E+06_9.81.txt",
#         "[0.5x0.01]_10_1E+06_9.81.txt"]

# names = ["0.3x0.01", "0.4x0.01", "0.5x0.01"]

# paths = ["[0.3x0.01]_10_1E+06_9.81.txt",
#         "[0.3x0.02]_10_1E+06_9.81.txt",
#         "[0.3x0.05]_10_1E+06_9.81.txt"]

# names = ["0.3x0.01", "0.3x0.02", "0.3x0.05"]

# paths = ["n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_50_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_55_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_60_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_65_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_70_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_75_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_85_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_90_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_95_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_100_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_110_230000.txt"]

# names = ["50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100", "110"]

paths = ["rnd/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
         "rnd/[0.3x0.01]_10_1E+06_9.81_1.0E-05_F_80_230000.txt"]

names = ["Random", "Grid-like"]

def readFile(path):
    times = []
    avgpeaks = []
    endpeaks = []
    with open(path, 'r') as f:
        for line in f:
            time, avgpeak, endpeak = line.split()
            times.append(float(time))
            avgpeaks.append(float(avgpeak))
            endpeaks.append(float(endpeak))

    return times, avgpeaks, endpeaks

# create figure with 2 plots
fig, axs = plt.subplots(1, 2)

for i, path in enumerate(paths):
    times, avgpeaks, endpeaks = readFile(path)

    top_times = times[::2]
    bot_times = times[1::2]
    middle_times = [(top + bot) / 2 for top, bot in zip(top_times, bot_times)]

    top_avgpeaks = avgpeaks[::2]
    bot_avgpeaks = avgpeaks[1::2]
    middle_avgpeaks = [(top + bot) / 2 for top, bot in zip(top_avgpeaks, bot_avgpeaks)]

    # plot in first plot
    axs[0].plot(middle_times, middle_avgpeaks, label=names[i])
    axs[0].set_xlabel('Time [s]', fontsize=12)
    axs[0].set_ylabel('Deflection [m]', fontsize=12)
    axs[0].set_title('Average distance of the bottom and top average peaks')
    axs[0].legend()

    top_endpeaks = endpeaks[::2]
    bot_endpeaks = endpeaks[1::2]
    middle_endpeaks = [(top + bot) / 2 for top, bot in zip(top_endpeaks, bot_endpeaks)]

    # plot in second plot
    axs[1].plot(middle_times, middle_endpeaks, label=names[i])
    axs[1].set_xlabel('Time [s]', fontsize=12)
    axs[1].set_ylabel('Deflection [cm]', fontsize=12)
    axs[1].set_title('Average distance of the bottom and top free end peaks')
    axs[1].legend()

# set width
fig.set_figwidth(12)

#set grid
axs[0].grid()
axs[1].grid()


plt.show()

# save figure
fig.savefig(f"stitched{names[1].replace(' ', '').replace('/', '').replace('^', '')}.png")


