import matplotlib.pyplot as plt

# Data from the log file
lon_ranges = [
    (-14.156859241940158, 144.00943250203338),  # gen 1
    (-67.14814885829009, 118.99176724312127),   # gen 1
    (-68.69522545361816, 111.82671075192431),   # gen 1
    (-125.57381799685213, 142.1723191046721),   # gen 1
    (-112.22164715731259, 161.46923062399932),  # gen 1
    (-78.90396284206236, 126.57045089061272),   # gen 1
    (-96.03138543174703, 138.0184136782104),    # gen 1
    (-39.31973004948928, 108.92145375664676),   # gen 2
    (-103.09434411085132, 73.87883007770687),   # gen 2
    (-102.13854994495978, 155.2954041537795),   # gen 2
    (-73.32044526158671, 79.92308889805216)     # gen 2
]

# Fitness values from the log file
fitness_values = [
    -14.359993,    # gen 1
    -0.4599947999999996,  # gen 1
    -0.6999966000000004,  # gen 1
    -3.0199938000000004,  # gen 1
    4.4599994999999995,   # gen 1
    1.620006,      # gen 1
    2.4000037,     # gen 1
    -0.5799953999999996,  # gen 2
    -1.7199948000000012,  # gen 2
    0.1800058,     # gen 2
    10.4800028     # gen 2
]

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plot points for generation 1
gen1_indices = range(7)  # First 7 points are from generation 1
plt.scatter([r[0] for r in lon_ranges[:7]], fitness_values[:7], 
           c='blue', label='Generation 1', alpha=0.6)

# Plot points for generation 2
gen2_indices = range(7, 11)  # Last 4 points are from generation 2
plt.scatter([r[0] for r in lon_ranges[7:]], fitness_values[7:], 
           c='red', label='Generation 2', alpha=0.6)

plt.xlabel("Longitude Range Minimum")
plt.ylabel("Fitness Value")
plt.title("Fitness Values vs Longitude Range Minimum by Generation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show() 