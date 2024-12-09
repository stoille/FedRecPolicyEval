import os
import json
import shutil

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

        def plot_group4(
                self,
                folder_path,
                x_param,
                x_label,
                metrics,
                titles,
                color_by,
                group_label,
                colors,
                show_raw=True,
                show_trendline=False,
                show_every_tenth_scatter=False,
                log_scale_metrics=None,
                truncate_y_to_x=False,
        ):
            # Load data
            graphs = self.load_data_from_folder(folder_path)
            if not graphs:
                print(f"No data loaded from {folder_path}.")
                return

            # Determine the number of graphs dynamically
            num_plots = min(len(metrics), len(titles))
            num_rows = (num_plots )  # Ensure enough rows for 2 columns
            fig, axes = plt.subplots(1,num_rows, figsize=(6 * num_rows, 6))
            axes = axes.flatten()[:num_plots]  # Adjust axes array to match the number of plots

            if log_scale_metrics is None:
                log_scale_metrics = []

            def plot_metric(ax, x_data, y_data, color, label, linestyle="-", fill=True, metric_name=None):
                """Helper function to plot raw values, trendline, and shaded region."""
                normalized = False  # Track whether data was normalized

                # Normalize y_data if not already probabilities
                if len(y_data) > 0 and (min(y_data) < 0 or max(y_data) > 1):
                    y_data = [y / max(y_data) if max(y_data) != 0 else 0 for y in y_data]
                    normalized = True

                # Adjust x_data for log scale if necessary
                if metric_name in log_scale_metrics:
                    ax.set_yscale('log')

                # Exclude the first value if the metric contains 'ut_norm'
                if metric_name == "train_ut_norm" or metric_name == "eval_ut_norm":
                    x_data = x_data[1:]
                    y_data = y_data[1:]

                # Truncate y_data to match x_data length if necessary
                if truncate_y_to_x and len(y_data) > len(x_data):
                    y_data = y_data[-len(x_data):]

                if len(x_data) != len(y_data):
                    print(f"Skipping plot for {metric_name}: mismatched lengths (x={len(x_data)}, y={len(y_data)})")
                    return

                # Update label to indicate normalization if applicable


                # Plot raw data
                if show_raw:
                    if show_every_tenth_scatter:
                        x_data = x_data[::10]
                        y_data = y_data[::10]
                    if len(x_data) > 0:
                        ax.scatter(x_data, y_data, color=color, s=50, zorder=3, label=label)
                    else:
                        ax.scatter([], [], color=color, s=50, zorder=3, label=label)

                # Plot trendline
                if show_trendline:
                    y_smoothed = gaussian_filter1d(y_data, sigma=2)
                    y_std = np.std(y_data) if len(y_data) > 1 else 0
                    ax.plot(x_data, y_smoothed, linestyle=linestyle, linewidth=2, color=color)
                    if fill:
                        ax.fill_between(x_data, y_smoothed - y_std, y_smoothed + y_std, alpha=0.2, color=color)

            for idx, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[idx]

                if isinstance(metric, set):  # Handle grouped metrics
                    sorted_metrics = sorted(metric)  # Sort grouped metrics
                    for sub_idx, sub_metric in enumerate(sorted_metrics):
                        linestyle = '-' if sub_idx == 0 else '--'  # Different styles for each sub-metric
                        for i, graph in enumerate(graphs):
                            label = f"{group_label}: {graph['config'].get(color_by, 'Unknown')} ({sub_metric})"
                            x_data = range(1, graph["config"].get(x_param, 0) + 1)
                            y_data = graph["metrics"].get(sub_metric, [])
                            if len(y_data) < len(x_data):  # Handle insufficient data
                                print(
                                    f"Error: Insufficient data for {sub_metric} in graph {i}. Expected {len(x_data)}, got {len(y_data)}.")
                                continue
                            plot_metric(ax, x_data, y_data, colors[i % len(colors)], label, linestyle,
                                        metric_name=sub_metric)
                else:  # Handle individual metrics
                    sorted_graphs = sorted(graphs,
                                           key=lambda g: g['config'].get(color_by, 'Unknown'))  # Sort by group_label
                    for i, graph in enumerate(sorted_graphs):
                        label = f"{group_label}: {graph['config'].get(color_by, 'Unknown')}"
                        x_data = range(1, graph["config"].get(x_param, 0) + 1)
                        y_data = graph["metrics"].get(metric, [])
                        if len(y_data) > len(x_data) and truncate_y_to_x:
                            y_data = y_data[-len(x_data):]
                        if len(y_data) < len(x_data):  # Handle insufficient data
                            print(
                                f"Error: Insufficient data for {metric} in graph {i}. Expected {len(x_data)}, got {len(y_data)}.")
                            continue
                        plot_metric(ax, x_data, y_data, colors[i % len(colors)], label, metric_name=metric)

                # Set titles and labels for each subplot
                ax.set_title(title, fontsize=16)
                ax.set_xlabel(x_label, fontsize=14)
                ax.set_ylabel(f"{metric} Values", fontsize=14)
                ax.grid(True)
                if idx == 0:  # Add legend only to the first subplot
                    ax.legend(fontsize=10)

            # Remove unused subplots if there are fewer metrics than subplots
            for idx in range(num_plots, len(axes)):
                fig.delaxes(axes[idx])

            plt.tight_layout()
            plt.show()

        def group_one(self):
            """Visualize metrics grouped by Beta."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run1",
                x_param="epochs",
                x_label="Time steps",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob", "train_correlated_mass"],
                titles=["Norm u", "Likable Probability", "Non-Likable Probability","Prob of Well-Corr Items" ],
                color_by="beta",
                group_label="β",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
                show_every_tenth_scatter=True,
                log_scale_metrics=["train_likable_prob", "train_nonlikable_prob"]
            )

        def group_two(self):
            """Visualize metrics grouped by Gamma."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run2",
                x_param="epochs",
                x_label="Time steps",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob","train_correlated_mass"],
                titles=["Norm u", "Likable Probability", "Non-Likable Probability","Prob of Well-Corr Items"],
                color_by="gamma",
                group_label="γ",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
            show_every_tenth_scatter = True,
            log_scale_metrics = ["train_likable_prob", "train_nonlikable_prob"],

            )

        def group_three(self):
            """Visualize metrics grouped by Number of Nodes."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run3",
                x_param="rounds",
                x_label="Federated rounds",
                metrics=["eval_ut_norm", "eval_likable_prob", "eval_nonlikable_prob","eval_correlated_mass"],
                titles=["Norm u", "Likable Probabilities", "Non-Likable Probabilities","Prob of Well-Corr Items"],
                color_by="num_nodes",
                group_label="N",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
                log_scale_metrics=["eval_likable_prob", "eval_nonlikable_prob"],
                truncate_y_to_x=False,

            )

        def group_three_pg(self):
            """Visualize metrics grouped by Number of Nodes."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run3",
                x_param="rounds",
                x_label="Federated rounds",
                metrics=[ "local_global_divergence",
                         "personalization_degree"],
                titles=[
                        "Local vs Global Divergence",
                        "Personalization Degree"],
                color_by="num_nodes",
                group_label="N",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
                truncate_y_to_x=False,

            )

        def group_three_ep(self):
            """Visualize metrics grouped by Number of Nodes."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run3",
                x_param="epochs",
                x_label="Time steps",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob","train_correlated_mass"],
                titles=["Norm u", "Likable Probabilities", "Non-Likable Probabilities","Prob of Well-Corr Items"],
                color_by="num_nodes",
                group_label="N",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
                log_scale_metrics=["train_likable_prob", "train_nonlikable_prob"],
                truncate_y_to_x=True,

            )

        def group_four(self):
            """Visualize metrics grouped by Epochs."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run4",
                x_param="rounds",
                x_label="Federated rounds",
                metrics=["eval_ut_norm", "eval_likable_prob", "eval_nonlikable_prob","eval_correlated_mass"],
                titles=["Norm u", "Likable Probabilities", "Non-Likable Probabilities","Prob of Well-Corr Items"],
                color_by="epochs",
                group_label="Epochs",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
                log_scale_metrics=["eval_likable_prob", "eval_nonlikable_prob"],
                truncate_y_to_x=False,

            )

        def group_four_pg(self):
            """Visualize metrics grouped by Epochs."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run4",
                x_param="rounds",
                x_label="Federated rounds",
                metrics=[ "local_global_divergence",
                         "personalization_degree"],
                titles=[ "Local vs Global Divergence",
                        "Personalization Degree"],
                color_by="epochs",
                group_label="Epochs",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
                truncate_y_to_x=False,

            )

        def group_four_ep(self):
            """Visualize metrics grouped by Epochs."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run4",
                x_param="epochs",
                x_label="Time steps",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob","train_correlated_mass"],
                titles=["Norm u", "Likable Probabilities", "Non-Likable Probabilities","Prob of Well-Corr Items"],
                color_by="epochs",
                group_label="Epochs",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
                log_scale_metrics=["train_likable_prob", "train_nonlikable_prob"],
                truncate_y_to_x=True,

            )

        # group one:
        # graph beta
        # x-axis = epochs
        # y-axis = eval_ut_norm, eval_likable_prob, eval_nonlikable_prob

        # group two:
        # graph gamma
        # x-axis = epochs
        # y-axis = eval_ut_norm, eval_likable_prob, eval_nonlikable_prob

        # group three:
        # num_nodes
        # x_achsis = traning rounds
        # y-achsis = eval_ut_norm, eval_likable_prob, eval_nonlikable_prob, local_global_divergence, personalization_degree

        # group four:
        # epochs
        # x_achsis = traning rounds
        # y-achsis = eval_ut_norm, eval_likable_prob, eval_nonlikable_prob, local_global_divergence, personalization_degree


        # group five:
        # fixed epochs
        # x_achsis = traning rounds
        # y-achsis = eval_ut_norm, eval_likable_prob, eval_nonlikable_prob, local_global_divergence, personalization_degree

        # group six Matrix of Plots 3 (lr_schedule) *4 (y-achsis)
        # num_nodes
        # x_achsis = traning rounds
        # y-achsis = eval_ut_norm, eval_likable_prob, eval_nonlikable_prob, local_global_divergence, personalization_degree
        # "lr_schedule": "constant", "decay" , Feature-specific

        def group_five(self):
            """Visualize metrics with fixed epochs, varying evaling rounds."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run5",
                x_param="rounds",
                x_label="Federated rounds",#Federated rounds
                metrics=["eval_ut_norm", "eval_likable_prob", "eval_nonlikable_prob","eval_correlated_mass"],
                titles=["Norm u", "Likable Probabilities", "Non-Likable Probabilities","Prob of Well-Corr Items"],
                color_by="rounds",
                group_label="Rounds",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
            log_scale_metrics = ["eval_likable_prob", "eval_nonlikable_prob"],
                truncate_y_to_x=False
            )

        def group_five_pg(self):
            """Visualize metrics with fixed epochs, varying evaling rounds."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run5",
                x_param="rounds",
                x_label="Federated rounds",#Federated rounds
                metrics=["local_global_divergence",
                         "personalization_degree"],
                titles=[ "Local vs Global Divergence",
                        "Personalization Degree"],
                color_by="rounds",
                group_label="Rounds",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
                truncate_y_to_x=False
            )

        def group_five_ep(self):
            """Visualize metrics with fixed epochs, varying evaling rounds."""
            self.plot_group4(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run5",
                x_param="epochs",
                x_label="Timesteps",  # Federated rounds
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob","train_correlated_mass"],
                titles=["Norm u", "Likable Probabilities", "Non-Likable Probabilities","Prob of Well-Corr Items"],
                color_by="rounds",
                group_label="Rounds",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
                log_scale_metrics=["train_likable_prob", "train_nonlikable_prob"],
                truncate_y_to_x=True
            )
        def group_six(self):
            self.plot_grouped_metrics(
                folder_path="/Users/work/PycharmProjects/FedRecPolicyEval/src/visuals/run6",
                x_param="rounds",
                x_label="Rounds",
                metrics=["train_ut_norm", "train_likable_prob", "train_nonlikable_prob", "local_global_divergence",
                         "personalization_degree"],
                conditions=["constant", "decay", "feature_specific"],
                condition_titles=["Constant", "Decay", "Feature-Specific"],
                titles=["Norm u", "Likable Probabilities", "Non-Likable Probabilities", "Local vs Global Divergence",
                        "Personalization Degree"],
                color_by="num_nodes",  # Compare based on number of nodes
                group_label="N",
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                show_raw=True,
                show_trendline=True,
            log_scale_metrics = ["train_likable_prob", "train_nonlikable_prob"],
                truncate_y_to_x=True,

            )

        def plot_grouped_metrics(
                self,
                folder_path,
                x_param,
                x_label,
                metrics,
                conditions,
                condition_titles,
                titles,
                color_by,
                group_label,
                colors,
                show_raw=True,
                show_trendline=False,
                show_every_tenth_scatter=False,
                log_scale_metrics=None,
                truncate_y_to_x=False  # New flag for truncating y_data
        ):
            graphs = self.load_data_from_folder(folder_path)
            if not graphs:
                print(f"No data loaded from {folder_path}.")
                return

            # Determine unique values of `color_by` to ensure consistent color mapping
            unique_color_values = sorted(set(g["config"].get(color_by) for g in graphs))
            color_map = {value: colors[idx % len(colors)] for idx, value in enumerate(unique_color_values)}

            # Determine grid dimensions: rows = conditions, columns = metrics
            num_conditions = len(conditions)
            num_metrics = len(metrics)
            fig, axes = plt.subplots(num_conditions, num_metrics, figsize=(6 * num_metrics, 6 * num_conditions))
            axes = np.array(axes).reshape(num_conditions, num_metrics)

            if log_scale_metrics is None:
                log_scale_metrics = []

            def plot_metric(ax, x_data, y_data, color, label, linestyle="-", fill=True, metric_name=None):
                """Helper function to plot raw values, trendline, and shaded region."""
                # Normalize y_data
                if len(y_data) > 0:
                    y_data = [y / max(y_data) if max(y_data) != 0 else 0 for y in y_data]

                # Adjust x_data for log scale if necessary
                if metric_name in log_scale_metrics:
                    ax.set_yscale('log')

                # Truncate y_data to match x_data length if necessary
                if truncate_y_to_x and len(y_data) > len(x_data):
                    y_data = y_data[-len(x_data):]

                if len(x_data) != len(y_data):
                    print(f"Skipping plot for {metric_name}: mismatched lengths (x={len(x_data)}, y={len(y_data)})")
                    return

                # Plot raw data
                if show_raw:
                    if show_every_tenth_scatter:
                        x_data = x_data[::10]
                        y_data = y_data[::10]
                    if len(x_data) > 0:
                        ax.scatter(x_data, y_data, color=color, s=50, zorder=3, label=label)
                    else:
                        ax.scatter([], [], color=color, s=50, zorder=3, label=label)

                # Plot trendline
                if show_trendline:
                    y_smoothed = gaussian_filter1d(y_data, sigma=2)
                    y_std = np.std(y_data) if len(y_data) > 1 else 0
                    ax.plot(x_data, y_smoothed, linestyle=linestyle, linewidth=2, color=color)
                    if fill:
                        ax.fill_between(x_data, y_smoothed - y_std, y_smoothed + y_std, alpha=0.2, color=color)

            used_labels = set()  # To prevent duplicate legend entries

            for condition_idx, condition in enumerate(conditions):
                condition_graphs = [g for g in graphs if g['config'].get('lr_schedule') == condition]

                if not condition_graphs:
                    print(f"No data found for condition: {condition}")
                    continue

                for metric_idx, metric in enumerate(metrics):
                    ax = axes[condition_idx, metric_idx]

                    if isinstance(metric, set):  # Handle grouped metrics
                        for sub_idx, sub_metric in enumerate(sorted(metric)):
                            linestyle = "-" if sub_idx == 0 else "--"  # Different styles for grouped metrics
                            for graph in condition_graphs:
                                x_data = range(1, graph["config"].get(x_param, 0) + 1)
                                y_data = graph["metrics"].get(sub_metric, [])
                                if len(y_data) < len(x_data):
                                    print(
                                        f"Error: Insufficient data for {sub_metric}. Expected {len(x_data)}, got {len(y_data)}.")
                                    continue

                                # Use the consistent color mapping
                                group_value = graph["config"].get(color_by, "Unknown")
                                color = color_map.get(group_value,
                                                      "#000000")  # Default to black if value not in color_map
                                label = f"{group_label}: {group_value} ({sub_metric})"

                                if label not in used_labels:
                                    plot_metric(ax, x_data, y_data, color, label, linestyle, metric_name=sub_metric)
                                    used_labels.add(label)
                                else:
                                    plot_metric(ax, x_data, y_data, color, None, linestyle, metric_name=sub_metric)

                    else:  # Handle individual metrics
                        for graph in condition_graphs:
                            x_data = range(1, graph["config"].get(x_param, 0) + 1)
                            y_data = graph["metrics"].get(metric, [])
                            if truncate_y_to_x and len(y_data) > len(x_data):
                                y_data = y_data[-len(x_data):]
                            if len(y_data) < len(x_data):
                                print(
                                    f"Error: Insufficient data for {metric}. Expected {len(x_data)}, got {len(y_data)}.")
                                continue

                            # Use the consistent color mapping
                            group_value = graph["config"].get(color_by, "Unknown")
                            color = color_map.get(group_value, "#000000")  # Default to black if value not in color_map
                            label = f"{group_label}: {group_value}"

                            if label not in used_labels:
                                plot_metric(ax, x_data, y_data, color, label, metric_name=metric)
                                used_labels.add(label)
                            else:
                                plot_metric(ax, x_data, y_data, color, None, metric_name=metric)

                    # Set subplot titles and labels
                    x_label_modified = f"log({x_label})" if metric in log_scale_metrics else x_label
                    ax.set_title(f"{titles[metric_idx]} ({condition_titles[condition_idx]})", fontsize=14)
                    ax.set_xlabel(x_label_modified, fontsize=12)
                    ax.set_ylabel(f"Normalized {metric}", fontsize=12)
                    ax.grid(True)

                    if condition_idx == 0 and metric_idx == 0:  # Add legend to the first subplot
                        ax.legend(fontsize=10)

            # Adjust layout for better readability
            plt.tight_layout()
            plt.show()

        def save(self):
            viz.group_one()
            viz.group_two()
            viz.group_three()
            viz.group_three_ep()
            viz.group_three_pg()
            viz.group_four()
            viz.group_four_ep()
            viz.group_four_pg()
            viz.group_five()
            viz.group_five_ep()
            viz.group_five_pg()


# Example usage
if __name__ == "__main__":
    viz = Visualization()
    viz.save()
