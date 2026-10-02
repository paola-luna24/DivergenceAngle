import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Parameters
d = 2 #in um
lamb = 1.3 #in um
n_II = 3.24 #index of refraction for core
n_I = 3.21 #index of refraction for cladding
b = np.linspace(0,1,500) #ranges from 0 to 1

# V-parameter
V = np.pi/lamb*2*d*np.sqrt(n_II**2-n_I**2)
LS = V*np.sqrt(1-b)


modes = []
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']

for m in range(1, 10):  # Check for modes m = 1, 2, 3, ...
    RS = np.arctan(np.sqrt(b / (1 - b))) + (m - 1) * np.pi / 2

    # Plot to visualize
    color = colors[m - 1]
    plt.plot(b, LS, color='black', linewidth=2, label='V$\sqrt{1-b}$' if m == 1 else '')
    plt.plot(b, RS, color=color, label=f'Right side (m={m})')

    # Find intersections by looking for sign changes
    difference = LS-RS
    sign_changes = np.where(np.diff(np.sign(difference)))[0]

    if len(sign_changes) > 0:
        modes.append(m)
    else:
        break  # No more modes

plt.xlabel('Effective index b (0,1)')
plt.ylabel('Solution')
# plt.title(f'Characteristic Equation (V = {V:.3f})')
plt.legend()
plt.grid(True)
plt.xlim(0, 1)
plt.show()

print(f"\nNumber of TE modes: {len(modes)}")
print(f"Modes exist for m = {modes}")
print(f"\nV: {V}")
