import numpy as np
import matplotlib.pyplot as plt


# paths = ["n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_45_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_50_230000.txt",
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
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_105_230000.txt",
#          "n/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_110_230000.txt"]

# names = ["45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95", "100", "105", "110"]

# paths = ["dt/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "dt/[0.3x0.01]_10_1E+06_9.81_2.0E-05_T_80_230000.txt",
#          "dt/[0.3x0.01]_10_1E+06_9.81_2.5E-06_T_80_230000.txt",
#          "dt/[0.3x0.01]_10_1E+06_9.81_5.0E-06_T_80_230000.txt",
#          "dt/[0.3x0.01]_10_1E+06_9.81_1.25E-06_T_80_230000.txt",]

# names = ["1.0E-05", "2.0E-05", "2.5E-06", "5.0E-06", "1.25E-06"]

# paths = ["ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_50000.txt",
#          "ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_80000.txt",
#          "ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_110000.txt",
#          "ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_140000.txt",
#          "ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_170000.txt",
#          "ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_200000.txt",
#          "ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_260000.txt",
#          "ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_290000.txt",
#          "ppm2/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_320000.txt"]

# names = ["50000", "80000", "110000", "140000", "170000", "200000", "230000", "260000", "290000", "320000"]

# paths = ["length/[0.2x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.225x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.25x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.275x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.325x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.35x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.375x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.4x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.425x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "length/[0.45x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",]

# names = ["0.2", "0.225", "0.25", "0.275", "0.3", "0.325", "0.35", "0.375", "0.4", "0.425", "0.45"]

# paths = ["height/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "height/[0.3x0.015]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "height/[0.3x0.02]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "height/[0.3x0.025]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "height/[0.3x0.03]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "height/[0.3x0.035]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "height/[0.3x0.04]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "height/[0.3x0.045]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "height/[0.3x0.05]_10_1E+06_9.81_1.0E-05_T_80_230000.txt"]

# names = ["0.01", "0.015", "0.02", "0.025", "0.03", "0.035", "0.04", "0.045", "0.05"]

# paths = ["density/[0.3x0.01]_2_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_5_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_15_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_20_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_25_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_30_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_35_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_40_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_45_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "density/[0.3x0.01]_50_1E+06_9.81_1.0E-05_T_80_230000.txt",]

# names = ["2", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50"]

paths = ["rnd/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
         "rnd/[0.3x0.01]_10_1E+06_9.81_1.0E-05_F_80_230000.txt",
         "rnd/[0.3x0.01]_10_2E+06_9.81_1.0E-05_T_80_230000.txt",
         "rnd/[0.3x0.01]_10_2E+06_9.81_1.0E-05_F_80_230000.txt",
         "rnd/[0.3x0.01]_10_4E+06_9.81_1.0E-05_T_80_230000.txt",
         "rnd/[0.3x0.01]_10_4E+06_9.81_1.0E-05_F_80_230000.txt",
         "rnd/[0.3x0.01]_10_6E+06_9.81_1.0E-05_T_80_230000.txt",
         "rnd/[0.3x0.01]_10_6E+06_9.81_1.0E-05_F_80_230000.txt",
         "rnd/[0.3x0.01]_10_8E+06_9.81_1.0E-05_T_80_230000.txt",
         "rnd/[0.3x0.01]_10_8E+06_9.81_1.0E-05_F_80_230000.txt"]

names = ["1E+06", "2E+06", "4E+06", "6E+06", "8E+06"]

# paths = ["force/[0.3x0.01]_10_1E+06_2.4525_1.0E-05_T_80_230000.txt",
#          "force/[0.3x0.01]_10_1E+06_4.905_1.0E-05_T_80_230000.txt",
#          "force/[0.3x0.01]_10_1E+06_9.81_1.0E-05_T_80_230000.txt",
#          "force/[0.3x0.01]_10_1E+06_14.715_1.0E-05_T_80_230000.txt",
#          "force/[0.3x0.01]_10_1E+06_19.62_1.0E-05_T_80_230000.txt",
#          "force/[0.3x0.01]_10_1E+06_24.525_1.0E-05_T_80_230000.txt",
#          "force/[0.3x0.01]_10_1E+06_29.43_1.0E-05_T_80_230000.txt",
#          "force/[0.3x0.01]_10_1E+06_34.335_1.0E-05_T_80_230000.txt"]

# names = ["2.4525", "4.905", "9.81", "14.715", "19.62", "24.525", "29.43", "34.335"]

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
fig, axs = plt.subplots()

for i, path in enumerate(paths):
    _, avgpeaks, endpeaks = readFile(path)

    dot1 = 'ro'
    dot2 = 'r>'

    if i % 2 == 1:
        dot1 = 'bo'
        dot2 = 'b>'

    n = float(names[i // 2])

    top_avgpeaks = avgpeaks[::2]
    bot_avgpeaks = avgpeaks[1::2]
    middle_avgpeaks = [(top + bot) / 2 for top, bot in zip(top_avgpeaks, bot_avgpeaks)]
    final_middle_avgpeak = middle_avgpeaks[-1]

    # plot final with n on x-axis
    axs.plot(n, final_middle_avgpeak, dot1)

    axs.set_xlabel('Tensile strength [Pa]', fontsize=12)
    axs.set_ylabel('Deflection [cm]', fontsize=12)
    axs.set_title('Final deflections of the average and free end vs force on \n\
the beam for both random and grid-like particle placement.')

    top_endpeaks = endpeaks[::2]
    bot_endpeaks = endpeaks[1::2]
    middle_endpeaks = [(top + bot) / 2 for top, bot in zip(top_endpeaks, bot_endpeaks)]
    final_middle_endpeak = middle_endpeaks[-1]

    # plot in second plot
    axs.plot(n, final_middle_endpeak, dot2)

# make x axis log
axs.set_yscale('log')
axs.legend(['Average Random', 'End Random', 'Average Grid-like', 'End Grid-like'])
# set grid
axs.grid()

plt.show()

# save figure
fig.savefig(f"stitched{names[1].replace(' ', '').replace('/', '').replace('^', '')}.png")


