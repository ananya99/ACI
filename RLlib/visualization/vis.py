import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = './RLlib/dummy_log.csv'  # <- replace with your filename

# Read the CSV into a DataFrame
df = pd.read_csv(csv_path)

# Check that the needed columns exist
required_cols = {'iteration', 'agent_0', 'adversary_0'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['agent_0'], marker='o', label='agent_0')
plt.plot(df['iteration'], df['adversary_0'], marker='s', label='adversary_0')

plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Agent_0 vs Adversary_0 over Iterations')
plt.legend()
plt.grid(True)

# Save the figure
output_path = './RLlib/dummy_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

# Show the plot (optional)
plt.show()