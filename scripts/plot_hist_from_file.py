import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(counts_0, counts_1, bins, normalize=False):
    plt.figure()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if normalize:
        counts_0 = counts_0 / np.sum(counts_0*np.diff(bins))
        counts_1 = counts_1 / np.sum(counts_1*np.diff(bins))
    plt.bar(bin_centers, counts_0, width=np.diff(bins), label="Label 0")
    plt.bar(bin_centers, counts_1, width=np.diff(bins), alpha=0.5, label="Label 1")
    plt.xlabel("Log Intensity")
    plt.legend()
    if normalize:
        plt.ylabel("Normalized Frequency")
        plt.title("Normalized Histogram of Intensities")
    else:
        plt.ylabel("Frequency")
        plt.title("Histogram of Intensities")
    plt.savefig("histogram.png")
    plt.close()

def main():
    data = np.load("histogram_data_94pb_99_percentile.npz")
    counts_0 = data["global_counts_0"]
    counts_1 = data["global_counts_1"]
    bins = data["bin_edges"]
    plot_histograms(counts_0, counts_1, bins, normalize=True)

if __name__ == "__main__":
    main()

