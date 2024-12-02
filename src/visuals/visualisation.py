import os
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


class Visualization:


        def load_data_from_folder(self, path: str):
            object_id = 1
            data = []
            if not os.path.exists(path):
                print(f"Path {path} does not exist.")
                return data

            os.chdir(path)
            for file in os.listdir():
                if file.endswith(".json"):
                    file_path = os.path.join(path, file)
                    with open(file_path, 'r') as f:
                        try:
                            file_content = json.load(f)
                            data.append(file_content)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON file: {file}")
                            continue
                    object_id += 1
            return data

        def plot_group(self, folder_path, x_param, x_label, metrics, titles, color_by, group_label, colors):
            graphs = self.load_data_from_folder(folder_path)
            if not graphs:
                print(f"No data loaded from {folder_path}.")
                return

            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            metrics_to_plot = [[metrics[0]], metrics[1:]]

            for idx, metric_group in enumerate(metrics_to_plot):
                ax = axes[idx]
                for i, graph in enumerate(graphs):
                    label = graph["config"][color_by]
                    x_data = range(1, graph["config"][x_param] + 1)

                    for metric_idx, metric in enumerate(metric_group):
                        y_data = graph["metrics"].get(metric, [])

                        if not y_data:  # Skip if data for the metric is empty or missing
                            print(f"Missing data for metric: {metric} in {folder_path}")
                            continue

                        # Smooth data using Gaussian filter
                        y_smoothed = gaussian_filter1d(y_data, sigma=2)

                        # Compute variability (std deviation)
                        y_std = np.std(y_data) if len(y_data) > 1 else 0

                        # Plot trendline
                        ax.plot(
                            x_data,
                            y_smoothed,
                            linestyle="-",
                            linewidth=2,
                            color=colors[i % len(colors)],
                            label=f"{group_label}: {label}" if metric_idx == 0 else None
                        )

                        # Add shaded area for std deviation
                        ax.fill_between(
                            x_data,
                            y_smoothed - y_std,
                            y_smoothed + y_std,
                            alpha=0.2,
                            color=colors[i % len(colors)]
                        )

                        # Plot raw data points every 10 units
                        real_points_x = x_data[::10]
                        real_points_y = y_data[::10]
                        ax.scatter(
                            real_points_x,
                            real_points_y,
                            color=colors[i % len(colors)],
                            s=50,
                            zorder=3,
                            label=None
                        )

                ax.set_title(titles[idx], fontsize=16)
                ax.set_xlabel(x_label, fontsize=14)
                ax.set_ylabel("", fontsize=14)
                ax.grid(True)
                if idx == 0:  # Add legend only to the first plot
                    ax.legend(fontsize=10)

            plt.tight_layout()
            plt.show()

        def group_one(self):
            """Visualize metrics grouped by Beta."""
            self.plot_group(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run1",
                x_param="epochs",
                x_label="Time steps / 10",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob"],
                titles=["Norm u", "Likable vs Non-Likable and Prob of Well-Corr. Items"],
                color_by="beta",
                group_label="β",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )

        def group_two(self):
            """Visualize metrics grouped by Gamma."""
            self.plot_group(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run2",
                x_param="epochs",
                x_label="Epochs",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob"],
                titles=["Norm u", "Likable vs Non-Likable and Prob of Well-Corr. Items"],
                color_by="gamma",
                group_label="γ",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )

        def group_three(self):
            """Visualize metrics grouped by Number of Nodes."""
            self.plot_group(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run3",
                x_param="epochs",
                x_label="Epochs",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob"],
                titles=["Norm u", "Likable vs Non-Likable and Prob of Well-Corr. Items"],
                color_by="num_nodes",
                group_label="N",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )

        def group_four(self):
            """Visualize metrics grouped by Epochs."""
            self.plot_group(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run4",
                x_param="rounds",
                x_label="Training Rounds",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob", "local_global_divergence",
                         "personalization_degree"],
                titles=["Norm u", "Likable vs Non-Likable and Other Metrics"],
                color_by="epochs",
                group_label="Epochs",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )

        # group one:
        # graph beta
        # x-axis = epochs
        # y-axis = train_ut_norm, train_likable_prob, train_nonlikable_prob

        # group two:
        # graph gamma
        # x-axis = epochs
        # y-axis = train_ut_norm, train_likable_prob, train_nonlikable_prob

        # group three:
        # num_nodes
        # x_achsis = traning rounds
        # y-achsis = train_ut_norm, train_likable_prob, train_nonlikable_prob, local_global_divergence, personalization_degree

        # group four:
        # epochs
        # x_achsis = traning rounds
        # y-achsis = train_ut_norm, train_likable_prob, train_nonlikable_prob, local_global_divergence, personalization_degree


        # group five:
        # fixed epochs
        # x_achsis = traning rounds
        # y-achsis = train_ut_norm, train_likable_prob, train_nonlikable_prob, local_global_divergence, personalization_degree

        # group six Matrix of Plots 3 (lr_schedule) *4 (y-achsis)
        # num_nodes
        # x_achsis = traning rounds
        # y-achsis = train_ut_norm, train_likable_prob, train_nonlikable_prob, local_global_divergence, personalization_degree
        # "lr_schedule": "constant", "decay" , Feature-specific

        def group_five(self):
            """Visualize metrics with fixed epochs, varying training rounds."""
            self.plot_group(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run5",
                x_param="rounds",
                x_label="Training Rounds",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob", "local_global_divergence",
                         "personalization_degree"],
                titles=["Norm u", "Likable vs Non-Likable and Other Metrics"],
                color_by="epochs",
                group_label="Epochs",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )

        def group_six(self):
            """Visualize metrics with a matrix of plots: 3 (lr_schedule) × 4 (metrics)."""
            # Load data
            folder_path = "/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run6"
            graphs = self.load_data_from_folder(folder_path)
            if not graphs:
                print(f"No data loaded from {folder_path}.")
                return

            # Define parameters
            lr_schedules = ["constant", "decay", "feature-specific"]
            metrics = ["train_ut_norm", "train_likable_prob", "train_nonlikable_prob", "local_global_divergence"]
            titles = ["Norm u", "Likable Probability", "Non-Likable Probability", "Local-Global Divergence"]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

            # Set up grid of subplots (3 lr_schedules × 4 metrics)
            fig, axes = plt.subplots(len(lr_schedules), len(metrics), figsize=(20, 12))
            for row, lr_schedule in enumerate(lr_schedules):
                for col, metric in enumerate(metrics):
                    ax = axes[row, col]
                    for i, graph in enumerate(graphs):
                        if graph["config"]["lr_schedule"] != lr_schedule:
                            continue  # Skip graphs that don't match the lr_schedule

                        label = graph["config"]["num_nodes"]  # Group by num_nodes
                        x_data = range(1, graph["config"]["rounds"] + 1)
                        y_data = graph["metrics"].get(metric, [])

                        if not y_data:
                            print(f"Missing data for metric: {metric} with lr_schedule: {lr_schedule}")
                            continue

                        # Smooth data using Gaussian filter
                        y_smoothed = gaussian_filter1d(y_data, sigma=2)

                        # Compute variability (std deviation)
                        y_std = np.std(y_data) if len(y_data) > 1 else 0

                        # Plot trendline
                        ax.plot(
                            x_data,
                            y_smoothed,
                            linestyle="-",
                            linewidth=2,
                            color=colors[i % len(colors)],
                            label=f"N: {label}" if row == 0 and col == 0 else None  # Single legend
                        )

                        # Add shaded area for std deviation
                        ax.fill_between(
                            x_data,
                            y_smoothed - y_std,
                            y_smoothed + y_std,
                            alpha=0.2,
                            color=colors[i % len(colors)]
                        )

                        # Plot raw data points every 10 units
                        real_points_x = x_data[::10]
                        real_points_y = y_data[::10]
                        ax.scatter(
                            real_points_x,
                            real_points_y,
                            color=colors[i % len(colors)],
                            s=50,
                            zorder=3,
                            label=None
                        )

                    # Add subplot titles and labels
                    ax.set_title(f"{titles[col]} ({lr_schedule})", fontsize=12)
                    ax.set_xlabel("Training Rounds", fontsize=10)
                    ax.set_ylabel("", fontsize=10)
                    ax.grid(True)

            # Add a single legend for all plots
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", fontsize=12, ncol=len(colors))
            plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])  # Adjust space for legend
            plt.show()

# Example usage
if __name__ == "__main__":
    viz = Visualization()
    viz.group_three()
