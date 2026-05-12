import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_offline_csv_comparison(csv_filepath):
    """
    Reads the offline CSV flight logs, averages the performance across all map seeds,
    and generates a 3-panel grouped bar chart comparing the two algorithms.
    """
    # 1. Load the database
    df = pd.read_csv(csv_filepath)
    
    # Optional but highly recommended: Filter out "Stuck" drones 
    # so they don't poison the average flight times and speeds
    #df_success = df[df['Final_State'] == 'Success']
    # 2. Identify the algorithms being compared (e.g., 'Normal' and 'Early')
    df_success = df
    algos = df_success['Algorithm'].unique()
    if len(algos) != 2:
        print(f"Warning: Expected exactly 2 algorithms, found {len(algos)}: {algos}")
        return
    name_a, name_b = algos[0], algos[1]
    
    # 3. Calculate Global Map Averages (Coverage)
    # Since coverage is duplicated across drone rows for a single map, drop duplicates first
    global_df = df_success[['Map_Seed', 'Algorithm', 'Coverage_pct']].drop_duplicates()
    global_stats = global_df.groupby('Algorithm')['Coverage_pct'].mean().to_dict()
    
    # 4. Calculate Per-Drone Averages
    # Group by Algorithm and Drone_ID, then calculate the mean across all Map Seeds
    drone_stats = df_success.groupby(['Algorithm', 'Drone_ID'])[['Speed_m_s', 'Jerk_m2_s5', 'Flight_Time_s']].mean().reset_index()
    
    # Split the data back into the two algorithms and sort to ensure Drone 0 to N align
    data_a = drone_stats[drone_stats['Algorithm'] == name_a].sort_values('Drone_ID')
    data_b = drone_stats[drone_stats['Algorithm'] == name_b].sort_values('Drone_ID')
    
    drone_ids = data_a['Drone_ID'].tolist()
    
    # ==========================================
    # PLOTTING
    # ==========================================
    x = np.arange(len(drone_ids))
    width = 0.35
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Global Title (Displays the Averaged Coverage)
    title_str = (f"Offline Trajectory Analysis: {name_a} vs {name_b} (Averaged across maps)\n"
                 f"Mean Coverage: {name_a} ({global_stats.get(name_a, 0):.2f}%) vs "
                 f"{name_b} ({global_stats.get(name_b, 0):.2f}%)")
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.95)
    
    color_a, color_b = '#2ca02c', '#1f77b4' # Green vs Blue
    
    # --- Subplot 1: Cornering Speed ---
    rects1_a = ax1.bar(x - width/2, data_a['Speed_m_s'], width, label=name_a, color=color_a, edgecolor='black')
    rects1_b = ax1.bar(x + width/2, data_b['Speed_m_s'], width, label=name_b, color=color_b, edgecolor='black')
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_title('Average Cornering Speed (Higher is usually more efficient)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(drone_ids)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Subplot 2: Cumulative Jerk ---
    rects2_a = ax2.bar(x - width/2, data_a['Jerk_m2_s5'], width, color=color_a, edgecolor='black')
    rects2_b = ax2.bar(x + width/2, data_b['Jerk_m2_s5'], width, color=color_b, edgecolor='black')
    ax2.set_ylabel('Jerk ($m^2/s^5$)')
    ax2.set_title('Average Cumulative Jerk (Lower means less actuator wear)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(drone_ids)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Subplot 3: Flight Time ---
    rects3_a = ax3.bar(x - width/2, data_a['Flight_Time_s'], width, color=color_a, edgecolor='black')
    rects3_b = ax3.bar(x + width/2, data_b['Flight_Time_s'], width, color=color_b, edgecolor='black')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Average Flight Time per Drone')
    ax3.set_xticks(x)
    ax3.set_xticklabels(drone_ids)
    
    # Dynamic Y-Limit to zoom in on the exact time range differences
    min_time = min(data_a['Flight_Time_s'].min(), data_b['Flight_Time_s'].min())
    max_time = max(data_a['Flight_Time_s'].max(), data_b['Flight_Time_s'].max())
    padding = (max_time - min_time) * 0.5 if max_time != min_time else 5
    ax3.set_ylim(max(0, min_time - padding), max_time + padding)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Utility: Add Data Labels ---
    def autolabel(rects, ax, format_str='{:.2f}'):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(format_str.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Apply labels
    autolabel(rects1_a, ax1)
    autolabel(rects1_b, ax1)
    autolabel(rects2_a, ax2, '{:.0f}')
    autolabel(rects2_b, ax2, '{:.0f}')
    autolabel(rects3_a, ax3)
    autolabel(rects3_b, ax3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

if __name__ == "__main__":
    filepath = "logs/switch_stats_seen_dist*2.csv"
    plot_offline_csv_comparison(filepath)