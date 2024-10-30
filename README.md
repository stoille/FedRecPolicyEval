---
tags: [basic, vision, fds]
dataset: [MovieLens]
framework: [torch, torchvision]
---

# Federated Recommender System Policy Evaluation with PyTorch and Flower

This project evaluates how recommender systems can implement user policy control in a federated way using the Flower framework. You may choose the recommender model to use either Matrix Factorization or Variational Autoencoder (VAE), conditioned on item metadata, user preferences, and user policies. Currently the [MovieLens](https://grouplens.org/datasets/movielens/) dataset is supported. Alternative datasets and implementations of policy-controlled recommender systems will be evaluated in the future.

## Set up the project

### Install dependencies and project

Install the dependencies defined in `pyproject.toml`. Simulation has been tested using Python 3.11.6.

```bash
pip install -e .
```
### Download MovieLens 1m

Download [MovieLens 1m](https://grouplens.org/datasets/movielens/1m/) and extract to ~/dev.

## Configure Model & Simulation Settings

Edit pyproject.toml. 

```bash
model-type = <mf|vae>
```

## Run the Project

You can run Flower projects such as this one in both _simulation_ and _deployment_ mode without making changes to the code. Start out by using the _simulation_ mode requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=5
```

### Run with the Deployment Engine

> \[!NOTE\]
> An update to this example will show how to run this Flower application with the Deployment Engine and TLS certificates, or with Docker.

### Visualize the results

Results will be automatically saved to `metrics_plots.png` and `latent_space_visualization.png` upon successful run.

