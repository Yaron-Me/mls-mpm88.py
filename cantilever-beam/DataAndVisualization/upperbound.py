import numpy as np
import matplotlib.pyplot as plt

# Given data
e_values = np.array([1e6, 2e6, 4e6, 6e6, 8e6, 1e7, 2e7, 4e7, 6e7, 8e7, 1e8, 2e8, 4e8, 6e8, 8e8, 1e9, 2e9, 4e9, 6e9, 8e9, 1e10, 2e10, 4e10, 6e10])
dt_values = np.array([2e-4, 1e-4, 1e-4, 9e-5, 8e-5, 7e-5, 5e-5, 3e-5, 3e-5, 2e-5, 2e-5, 1e-5, 1e-5, 9e-6, 8e-6, 7e-6, 5e-6, 3e-6, 3e-6, 2e-6, 2e-6, 1.5e-6, 1.2e-6, 1e-6])
other_dt = np.array([2.00E-04, 1.00E-04, 1.00E-04, 8.00E-05, 7.00E-05, 6.00E-05, 4.00E-05, 3.00E-05, 2.00E-05, 2.00E-05, 2.00E-05, 1.00E-05, 1.00E-05, 8.00E-06, 7.00E-06, 6.00E-06, 4.00E-06, 3.00E-06, 2.00E-06, 2.00E-06, 2.00E-06, 1.25E-06, 1.00E-06, 8.50E-07])

#-----------------------
# Perform linear regression on log-log data
log_e = np.log10(e_values)
log_dt = np.log10(dt_values)
coefficients = np.polyfit(log_e, log_dt, 1)  # Fit a line to log-log data
slope, intercept = coefficients

# Generate fitted line
log_e_fit = np.linspace(log_e.min(), log_e.max(), 100)
log_dt_fit = slope * log_e_fit + intercept

# Convert fitted line back to original scale
e_fit = 10**log_e_fit
dt_fit = 10**log_dt_fit
#-----------------------
loog_dt = np.log10(other_dt)
coefficients = np.polyfit(log_e, loog_dt, 1)
sloope, interceept = coefficients
loog_dt_fit = sloope * log_e_fit + interceept
dt_fiit = 10**loog_dt_fit



# Plot the data and the fit
plt.figure()
plt.loglog(e_values, dt_values, 'ro', label='0.05 m height data')
plt.loglog(e_fit, dt_fit, 'r:', label=f'0.05 m Fit: dt = {10**intercept:.2e} * exp({slope:.2f}E)')
plt.loglog(e_values, other_dt, 'bo', label='0.01 m height data')
plt.loglog(e_fit, dt_fiit, 'b:', label=f'0.01 m Fit: dt = {10**interceept:.2e} * exp({sloope:.2f}E)')

plt.rc('font', size=12)

plt.xlabel(r'$E$ in [Pa]', fontsize=14)
plt.ylabel(r'd$t$ in [s]', fontsize=14)
plt.title(r'Highest stable d$t$ versus $E$ with Power Law Fit', fontsize=14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Print the coefficients
print(f"Slope (n): {slope}")
print(f"Intercept (log10(k)): {intercept}")
print(f"Intercept (k): {10**intercept}")
