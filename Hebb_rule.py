# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
#     format_version: '1.3'
# ---

# %% [markdown]
# # Hebbian Learning — AND Gate (Jupytext / percent format)
#
# This file demonstrates a simple Hebbian learning rule on a bipolar AND gate.
# It includes:
# - Hebbian training function
# - Prediction function
# - Activation (step) plot
# - Decision boundary plot **before** and **after** training
#
# Save this file as `Hebbian_Learning_percent.py`. GitHub will show it as a code file;
# if you use **Jupytext** it will open as a notebook (markdown + code cells) and render nicely.

# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# %% [markdown]
# ## Activation function (step / bipolar)

# %%
def activation_function(x):
    """Step activation: returns +1 for x >= 0, else -1"""
    return 1 if x >= 0 else -1

# %% [markdown]
# ## Hebbian training
#
# We use bipolar inputs `{-1, +1}` and targets `{-1, +1}`.  
# We keep weights in the form: `[bias, w1, w2]`
# (bias is treated as weight for a constant input of 1).

# %%
def hebb_train(inputs, targets, init_weights=None, verbose=False):
    """
    Train using Hebbian rule (fast/one-shot updates).
    inputs: (n_samples, n_features) array with bipolar values (-1 or +1)
    targets: (n_samples,) array with bipolar values (-1 or +1)
    init_weights: optional initial weights (bias + n_features)
    Returns: weights as np.array([bias, w1, w2, ...])
    """
    n_features = inputs.shape[1]
    if init_weights is None:
        weights = np.ones(n_features + 1, dtype=float)  # bias + weights
    else:
        weights = init_weights.astype(float).copy()

    if verbose:
        print("Initial weights:", weights)

    for x, y in zip(inputs, targets):
        x_aug = np.array([1.0] + list(x))  # add bias input
        weights = weights + y * x_aug      # Hebbian update: W += y * x_aug
        if verbose:
            print(f"Input: {x}, Target: {y}, Updated weights: {weights}")

    return weights

# %% [markdown]
# ## Prediction (inference)

# %%
def predict(weights, x):
    """
    Predict label (+1 or -1) for input x (1D array of features).
    weights: [bias, w1, w2, ...]
    """
    x_aug = np.array([1.0] + list(x))
    net = np.dot(weights, x_aug)
    return activation_function(net)

# %% [markdown]
# ## Plot: Activation (step) function

# %%
def plot_activation(save_path=None):
    xs = np.linspace(-5, 5, 500)
    ys = np.array([activation_function(v) for v in xs])

    plt.figure(figsize=(6, 3.5))
    plt.plot(xs, ys, linewidth=2)
    plt.axvline(0, linewidth=1, linestyle='--')
    plt.title("Activation Function (Step) — bipolar")
    plt.xlabel("Net input")
    plt.ylabel("Output")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

# %% [markdown]
# ## Plot: Decision boundary
#
# For weights `[b, w1, w2]` we draw the line `w1*x + w2*y + b = 0`.
# Handle case when w2 is nearly zero (vertical boundary).

# %%
def plot_decision_boundary(weights, inputs, targets, title=None, save_path=None):
    """
    Plot input points and the decision boundary defined by weights.
    inputs: (n_samples, 2)
    targets: (n_samples,) in {-1, +1}
    """
    b = weights[0]
    w1 = weights[1]
    w2 = weights[2]

    xs = np.linspace(-2.0, 2.0, 400)

    plt.figure(figsize=(6, 6))
    # If w2 is small, plot vertical line x = -b/w1
    eps = 1e-9
    if abs(w2) < eps and abs(w1) > eps:
        x_vert = -b / w1
        plt.axvline(x_vert, linewidth=2, label="Decision boundary")
    elif abs(w1) < eps and abs(w2) > eps:
        y_horiz = -b / w2
        plt.axhline(y_horiz, linewidth=2, label="Decision boundary")
    else:
        ys = -(b + w1 * xs) / (w2 + eps)
        plt.plot(xs, ys, linewidth=2, label="Decision boundary")

    # Plot datapoints
    # Map targets (+1) -> marker '^', (-1) -> 'o' and color-coded using c
    for p, t in zip(inputs, targets):
        marker = '^' if t == 1 else 'o'
        plt.scatter(p[0], p[1], s=140, marker=marker)
        plt.text(p[0] + 0.06, p[1] + 0.06, f"{t}", fontsize=10)

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

# %% [markdown]
# ## Example dataset — Bipolar AND gate
#
# Inputs are bipolar: +1 represents logical 1, -1 represents logical 0 (bipolar encoding)

# %%
inputs = np.array([
    [ 1,  1],
    [ 1, -1],
    [-1,  1],
    [-1, -1]
], dtype=float)

targets = np.array([1, -1, -1, -1], dtype=float)  # AND gate (bipolar)

# %% [markdown]
# ## Run demo (training, plotting, testing)
#
# This block runs only when the file is executed (not when imported).

# %%
def demo(save_plots=True, verbose=True):
    out_dir = "plots"
    if save_plots and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initial weights (bias + w1 + w2) initialized to ones
    weights_init = np.ones(inputs.shape[1] + 1, dtype=float)

    if verbose:
        print("Initial weights:", weights_init)

    # Plot activation function
    act_path = os.path.join(out_dir, "activation.png") if save_plots else None
    plot_activation(save_path=act_path)
    if verbose and save_plots:
        print("Saved activation plot to:", act_path)

    # Plot decision boundary before training
    before_path = os.path.join(out_dir, "before.png") if save_plots else None
    plot_decision_boundary(weights_init, inputs, targets, title="Before training (initial weights)", save_path=before_path)
    if verbose and save_plots:
        print("Saved BEFORE-training decision boundary to:", before_path)

    # Train
    weights_final = hebb_train(inputs, targets, init_weights=weights_init, verbose=verbose)
    if verbose:
        print("\nFinal weights after Hebbian training:", weights_final)

    # Plot decision boundary after training
    after_path = os.path.join(out_dir, "after.png") if save_plots else None
    plot_decision_boundary(weights_final, inputs, targets, title="After training (Hebbian learned)", save_path=after_path)
    if verbose and save_plots:
        print("Saved AFTER-training decision boundary to:", after_path)

    # Test predictions
    if verbose:
        print("\nPredictions on training set:")
        for x in inputs:
            print(f" Input {list(x)} -> {predict(weights_final, x)}")

    return weights_init, weights_final

# %% [markdown]
# ## If run as a script, execute demo

# %%
if __name__ == "__main__":
    demo(save_plots=True, verbose=True)
