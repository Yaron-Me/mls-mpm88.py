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

import matplotlib.pyplot as plt

times, avgpeaks, endpeaks = readFile("[0.3x0.01]_10_1E+06_9.81.txt")

top_times = times[::2]
bot_times = times[1::2]

top_avgpeaks = avgpeaks[::2]
bot_avgpeaks = avgpeaks[1::2]

top_endpeaks = endpeaks[::2]
bot_endpeaks = endpeaks[1::2]


fig, axs = plt.subplots()

# plot avgpeaks as red line one normal one dotted
axs.plot(top_times, top_avgpeaks, 'r-', label='Top oscillation of average')
axs.plot(bot_times, bot_avgpeaks, 'r:', label='Bottom oscillation of average')

# plot endpeaks as blue line one normal one dotted
axs.plot(top_times, top_endpeaks, 'b-', label='Top oscillation of free end')
axs.plot(bot_times, bot_endpeaks, 'b:', label='Bottom oscillation of free end')

# set labels
axs.set_xlabel('Time [s]', fontsize=12)
axs.set_ylabel('Deflection [cm]', fontsize=12)
axs.set_title('Distance from expected deflection over time of oscillations')

plt.legend()
plt.show()
