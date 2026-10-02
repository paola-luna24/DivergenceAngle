from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erf
from scipy.optimize import minimize, curve_fit
import matplotlib.image as mpimg

def beam_profile(c, x, background=0, forwards=True):
    # Unpack coefficients
    a, x0, w = c[0], c[1], c[2]

    # Profile based on coeficcients
    return (a * 0.5 * (1 - erf(np.sqrt(2) * (x - x0) / w))) + background

def error_function(c, x, background, data):
    guess = beam_profile(c, x, background)

    return np.sum((guess - data) ** 2)

def linear_far_field(z, theta, z0):
    return theta * (z - z0)

def find_beam_width(x, data, background=0, plot=False):
    # Flip data if it is in the wrong order
    if data[0] < data[-1]:
        data = np.flip(data)
    # Find optimal parameters to minimize error function
    guess = np.array([3, 12, 10]) # (a, x0, width)
    opt = minimize(error_function, x0=guess, args=(x, background, data))
    # Extract the optimal coefficients
    c = opt['x']
    # print(c)
    # print(f'beam width = {c[2]:.4f} mm')
    fit = beam_profile(c, x, background)
    # Plot to compare
    if plot:
        fig, ax = plt.subplots(1)
        ax.plot(x, data)
        ax.plot(x, fit)
        # Display the plot
        plt.show()
    # Return the width and fit
    return c[2], fit

def process_csv_file(csv_filepath, background=0.00059, plot=False):
    # 1. Read the CSV file into Pandas
    df = pd.read_csv(csv_filepath)

    widths = []
    fits = []
    file_label = Path(csv_filepath).stem

    # 2. Extract position and compute widths for each submeasurement
    x = df["Pos_1"].values

    for i in range(1, 5):
        sub_x = df[f"Pos_{i}"].values
        sub_data = df[f"Power_{i}"].values

        width, fit = find_beam_width(sub_x, sub_data, background=background)
        widths.append(width)
        fits.append(fit)

        print(f"[{file_label} - Sub {i}] Beam Width = {width:.4f} mm")

    # 3. Calculate mean power across all 4 submeasurements and fit the mean curve
    power_cols = [f"Power_{i}" for i in range(1, 5)]
    mean_power = df[power_cols].mean(axis=1).values

    avg_width, avg_fit = find_beam_width(x, mean_power, background=background)

    # 4. Single plot matching the reference style
    if plot:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Plot averaged raw data as blue points
        ax.plot(
            x,
            mean_power,
            "o",
            color="black",
            markersize=5
        )

        # Smooth the fitted curve for display
        x_smooth = np.linspace(0, x.max(), 300)
        # Refit on mean data to obtain smooth line parameters
        opt_mean = minimize(
            error_function,
            x0=[3, 12, 10],
            args=(x, background, mean_power),
        )
        fit_smooth = beam_profile(opt_mean["x"], x_smooth, background)

        # Plot single fitted curve as a thick red line
        ax.plot(
            x_smooth,
            fit_smooth,
            "-",
            color="red",
            linewidth=2.5,
            label=f"Fit (w = {avg_width:.2f} mm)",
        )

        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['font.family'] = 'sans-serif'

        ax.grid(True, which='major', color='#888888', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        # Match formatting from the target image
        ax.set_xlabel("Razorblade Position (mm)", fontweight='bold', fontsize=20)
        ax.set_ylabel("Measured Power (mW)", fontweight='bold', fontsize=20)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        # Set size of numbers on axis
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(left=x.min() - 0.5, right=x.max() + 0.5)

        plt.tight_layout()
        plt.savefig(
            f"{file_label}_avg_fit.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    return widths, fits

# Gaussian Beam Model
def gaussian_beam_width(z, w0, zR, z0):
    return w0 * np.sqrt(1 + ((z - z0) / zR) ** 2)


plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# Raw dataset 1
dataset1_raw = [
    [6, [[6.10911, 0.251065], [6.07274, 0.131147], [6.05883, 0.112086], [5.85047, 0.176947]]],
    [9, [[6.47891, 0.143509], [6.39970, 0.129427], [6.44376, 0.163070], [6.46323, 0.111101]]],
    [12, [[6.74379, 0.249912], [6.86838, 0.201311], [6.71284, 0.212115], [6.66258, 0.328515]]],
    [15, [[7.37393, 0.191489], [7.38084, 0.265283], [7.27366, 0.194843], [7.38589, 0.281997]]],
    [18, [[7.56050, 0.204325], [7.61364, 0.211701], [7.66441, 0.231034], [7.81051, 0.256085]]],
    [21, [[7.96437, 0.214078], [7.95715, 0.868230], [8.53750, 0.198489], [7.93344, 0.227115]]],
    [24, [[8.59451, 0.319100], [8.34335, 0.306890], [8.53367, 0.274206], [8.69821, 0.242636]]],
    [27, [[8.83348, 0.411532], [9.37056, 0.282323], [8.94694, 0.333910], [8.86416, 0.368869]]],
    [30, [[9.49112, 0.516806], [9.28804, 0.474332], [9.16913, 0.467386], [9.35019, 0.405741]]]
]

# Raw dataset 2
dataset2_raw = [
    [6, [[6.60078, 0.104009], [6.69612, 0.0775226]]],
    [9, [[7.14436, 0.114934], [7.07390, 0.105673]]],
    [12, [[7.27947, 0.146485], [7.59480, 0.104985]]],
    [15, [[8.14189, 0.0965097], [7.95053, 0.109176]]],
    [18, [[8.51562, 0.0873398], [8.62666, 0.117104]]],
    [21, [[9.51419, 0.482102], [8.92264, 0.179623]]],
    [24, [[9.48793, 0.145628], [9.59121, 0.147731]]],
    [27, [[9.79079, 0.227118], [10.0675, 0.157292]]],
    [30, [[10.8319, 0.205857], [10.6854, 0.270142]]]
]

def linear_far_field(z, theta, z0):
    return theta * (z - z0)


def process_raw_dataset(raw_data):
    # Calculate weighted means and positions
    z_positions_mm, weighted_means, weighted_sigmas = [], [], []

    for entry in raw_data:
        pos_cm = entry[0]
        sub_pairs = np.array(entry[1])

        values = sub_pairs[:, 0]
        uncertainties = sub_pairs[:, 1]

        # Inverse-variance weighting
        weights = 1.0 / (uncertainties ** 2)
        mean = np.sum(values * weights) / np.sum(weights)
        sigma = np.sqrt(1.0 / np.sum(weights))

        z_positions_mm.append(pos_cm * 10.0)  # Convert cm to mm
        weighted_means.append(mean)
        weighted_sigmas.append(sigma)

    return np.array(z_positions_mm), np.array(weighted_means), np.array(weighted_sigmas)


# Process both datasets
z1_mm, y1_mean, y1_sigma = process_raw_dataset(dataset1_raw)
z2_mm, y2_mean, y2_sigma = process_raw_dataset(dataset2_raw)

# Fit Dataset 1
popt1, pcov1 = curve_fit(linear_far_field, z1_mm, y1_mean, sigma=y1_sigma, absolute_sigma=True, p0=[0.014, -300.0])
perr1 = np.sqrt(np.diag(pcov1))

# Fit Dataset 2
popt2, pcov2 = curve_fit(linear_far_field, z2_mm, y2_mean, sigma=y2_sigma, absolute_sigma=True, p0=[0.014, -300.0])
perr2 = np.sqrt(np.diag(pcov2))

d1_mean, d1_sigma = y1_mean * 2, y1_sigma * 2
d2_mean, d2_sigma = y2_mean * 2, y2_sigma * 2

print(f"Dataset 1 -> θ: {popt1[0]:.6f} ± {perr1[0]:.6f} rad | z0: {popt1[1]:.3f} ± {perr1[1]:.3f} mm")
print(f"Dataset 2 -> θ: {popt2[0]:.6f} ± {perr2[0]:.6f} rad | z0: {popt2[1]:.3f} ± {perr2[1]:.3f} mm")

# Plotting both together
fig, ax = plt.subplots(figsize=(8, 5.15))
z_smooth = np.linspace(min(0, z2_mm.min()) - 10, max(z1_mm.max(), z2_mm.max()) + 10, 200)

# Plot dataset 1
ax.errorbar(
    z1_mm, d1_mean, yerr=d1_sigma, fmt='o', color='blue',
    ecolor='blue', elinewidth=1.0, capsize=3, markersize=5,
    label='-X̂ cage', zorder=3
)
ax.plot(z_smooth, 2* linear_far_field(z_smooth, *popt1), color='blue', linestyle='-', linewidth=1.8,
        label=f'θ=14.1 mrad', zorder=2)

#  Plot dataset 2
ax.errorbar(
    z2_mm, d2_mean, yerr=d2_sigma, fmt='o', color='red',
    ecolor='red', elinewidth=1.0, capsize=3, markersize=5,
    label='-Ŷ-Ẑ cage', zorder=3
)
ax.plot(z_smooth, 2* linear_far_field(z_smooth, *popt2), color='red', linestyle='-', linewidth=1.8,
        # label=f'θ={popt2[0] * 1000:.2f} mrad', zorder=2)
        label=f'θ=16.9 mrad', zorder=2)


# Formatting the plot
ax.grid(True, color='#888888', linestyle='-', linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)
ax.set_xlabel("Position along optical axis (mm)", fontsize=24, fontweight='bold')
ax.set_ylabel("Gaussian 1/e$^2$ diameter (mm)", fontsize=24, fontweight='bold')
ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=16)
ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.01, 0.84), # (x, y) coordinates relative to plot area
    frameon=True,
    prop={'weight': 'bold', 'size': 18}
)


plt.tight_layout()
plt.savefig("Paper_gaussian_fits.png", dpi=300, bbox_inches='tight')
# plt.show()


# Create the figure and axes
fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

# Retrieve images
image1 = mpimg.imread('Paper_data2_6cm_avg_fit.png')
image2 = mpimg.imread('Paper_gaussian_fits.png')


fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # width, height in inches

axes[0].imshow(image1)
axes[0].axis("off")

axes[1].imshow(image2)
axes[1].axis("off")

axes[0].imshow(image1)
axes[0].axis("off")
axes[0].text(
    0.1, 0.92, "(a)",
    transform=axes[0].transAxes,
    fontsize=14,
    fontweight="bold",
    color="black",
    va="top",
    ha="left"
)

axes[1].imshow(image2)
axes[1].axis("off")
axes[1].text(
    0.12, 0.92, "(b)",
    transform=axes[1].transAxes,
    fontsize=14,
    fontweight="bold",
    color="black",
    va="top",
    ha="left"
)

plt.tight_layout()
plt.savefig("Diver_angle.svg", dpi=300, bbox_inches='tight')
plt.show()

