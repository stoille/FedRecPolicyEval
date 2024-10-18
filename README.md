---
tags: [basic, vision, fds]
dataset: [MovieLens]
framework: [torch, torchvision]
---

# Federated Recommender System Policy Evaluation with PyTorch and Flower

This project evaluates how recommender systems can implement user policy control in a federated way using the Flower framework. The codebase is extended from [FedVAE](https://github.com/adap/flower/tree/main/examples/pytorch-federated-variational-autoencoder), but uses the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. The default recommender system is based on a variational autoencoder (VAE) that has been conditioned on item metadata, user preferences, and user policies. Alternative implementations of policy-controlled recommender systems will be evaluated in the future.

## Set up the project

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fedvaeexample` package.

```bash
pip install -e .
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

```bash
python src/plot_results.py
```

