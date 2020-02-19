import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_signal(data):
    sig = data["signals"]["Chin"]
    n_seconds, sampling_rate = sig.shape
    fig, ax = plt.subplots(1, figsize=(14, 10))
    min_point, max_point = 0, 0
    x_time_axis = [i / sampling_rate for i in range(n_seconds * sampling_rate)]
    min_point = min(min_point, sig.min())
    max_point = max(max_point, sig.max())
    ax.plot(x_time_axis, sig.ravel(), label="Chin")
    ax.set_xlabel("seconds")
    ax.set_ylabel("signal")
    ax.legend()

    # Add event annotations
    events = data["rswa_events"]
    if events:
        for start_time, end_time, etype in events:
            start_time -= data["staging"][0][0]
            end_time -= data["staging"][-1][0]
            if etype == "P":
                ax.axvline(x=start_time, linestyle="--", color="black")

            elif etype == "T":
                x_start = start_time
                y_start = min_point
                width = float(end_time - start_time)
                height = max_point - min_point
                rect = Rectangle((x_start, y_start), width, height, 
                                 linewidth=1, fill=True, alpha=0.5, 
                                 facecolor="grey")
                ax.add_patch(rect)
            else:
            	continue
    plt.show()
