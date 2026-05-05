import matplotlib.pyplot as plt
import numpy as np

# --- 1. THE DATA ---
drones = ['Drone 0', 'Drone 1', 'Drone 2', 'Drone 3', 'Drone 4']

# Early Switching Data
early_speed = [1.73, 1.52, 1.66, 1.63, 1.47]
early_jerk = [2359.60, 2275.06, 1300.87, 2212.01, 1883.70]
early_miss = [4.01, 4.01, 3.51, 3.98, 3.99]

# Normal Switching Data
normal_speed = [1.37, 1.19, 1.42, 1.33, 1.24]
normal_jerk = [2338.85, 2268.19, 1263.63, 2203.73, 1931.99]
normal_miss = [4.00, 4.00, 3.51, 3.98, 3.98]

# Global Metrics (for the title)
early_time, normal_time = 37.20, 39.75
early_cov, normal_cov = 98.92, 99.07

# --- 2. PLOT SETUP ---
x = np.arange(len(drones))  # the label locations
width = 0.35  # the width of the bars

# Create a figure with 3 subplots stacked vertically
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Global Title
fig.suptitle('Trajectory Tracking Performance: Early vs Normal Switching\n'
             f'Total Time: Early ({early_time}s) vs Normal ({normal_time}s) | '
             f'Coverage: Early ({early_cov}%) vs Normal ({normal_cov}%)', 
             fontsize=14, fontweight='bold', y=0.95)

# --- Subplot 1: Cornering Speed (Higher is better for momentum) ---
rects1_e = ax1.bar(x - width/2, early_speed, width, label='Early Switching', color='#2ca02c', edgecolor='black')
rects1_n = ax1.bar(x + width/2, normal_speed, width, label='Normal Switching', color='#1f77b4', edgecolor='black')
ax1.set_ylabel('Speed (m/s)')
ax1.set_title('Average Cornering Speed (Kinetic Energy Retention)')
ax1.set_xticks(x)
ax1.set_xticklabels(drones)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# --- Subplot 2: Cumulative Jerk ---
rects2_e = ax2.bar(x - width/2, early_jerk, width, color='#2ca02c', edgecolor='black')
rects2_n = ax2.bar(x + width/2, normal_jerk, width, color='#1f77b4', edgecolor='black')
ax2.set_ylabel('Jerk ($m^2/s^5$)')
ax2.set_title('Cumulative Jerk Effort (Actuator Wear)')
ax2.set_xticks(x)
ax2.set_xticklabels(drones)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# --- Subplot 3: Miss Distance (Lower is more precise) ---
# We use a custom Y-limit here because the differences are tiny
rects3_e = ax3.bar(x - width/2, early_miss, width, color='#2ca02c', edgecolor='black')
rects3_n = ax3.bar(x + width/2, normal_miss, width, color='#1f77b4', edgecolor='black')
ax3.set_ylabel('Distance (m)')
ax3.set_title('Average Miss Distance (Coverage Trade-off)')
ax3.set_xticks(x)
ax3.set_xticklabels(drones)
ax3.set_ylim(3.0, 4.5) # Zoom in to see the millimeter differences
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# --- Add Data Labels ---
def autolabel(rects, ax, format_str='{:.2f}'):
    """Attach a text label above each bar."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(format_str.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# Apply labels to all bars
autolabel(rects1_e, ax1); autolabel(rects1_n, ax1)
autolabel(rects2_e, ax2, '{:.0f}'); autolabel(rects2_n, ax2, '{:.0f}')
autolabel(rects3_e, ax3); autolabel(rects3_n, ax3)

plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust layout so title isn't clipped 
plt.show(block=True)