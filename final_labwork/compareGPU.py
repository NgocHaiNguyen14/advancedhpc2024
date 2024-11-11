import matplotlib.pyplot as plt

# Execution times
labels = ['GPU with Shared Memory', 'GPU without Shared Memory']
times = [1.0237548351287842, 1.1964266300201416]
plt.figure(figsize=(8, 5))
plt.bar(labels, times, color=['skyblue', 'salmon'])
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison: GPU with vs without Shared Memory')
for i, time in enumerate(times):
    plt.text(i, time + 0.01, f"{time:.3f}", ha='center', va='bottom', fontsize=10)

# Show the plot
plt.show()
