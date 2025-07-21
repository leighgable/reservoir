import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random, lax

# prompt: # Function for initializing a matrix as a Barabasi-Albert scale-free network using jax.numpy


def barabasi_albert_matrix(n_nodes, m, seed=None):
  """
  Initializes an adjacency matrix as a Barabasi-Albert scale-free network.

  Args:
    n_nodes (int): The number of nodes in the network.
    m (int): The number of edges to attach from a new node to existing nodes.
    seed (int, optional): Random seed for reproducibility. Defaults to None.

  Returns:
    jax.numpy.ndarray: The adjacency matrix of the Barabasi-Albert network.
  """
  if seed is not None:
    random.seed(seed)

  # Initialize with m fully connected nodes
  adj_matrix = jnp.zeros((n_nodes, n_nodes), dtype=int)
  for i in range(m):
    for j in range(m):
      if i != j:
        adj_matrix = adj_matrix.at[i, j].set(1)

  # Add remaining nodes
  for i in range(m, n_nodes):
    # Select m existing nodes based on their degree
    degrees = jnp.sum(adj_matrix[:i, :i], axis=1)
    probabilities = degrees / jnp.sum(degrees)
    connected_nodes = random.choices(range(i), weights=probabilities, k=m)

    # Add edges to the new node
    for connected_node in connected_nodes:
      adj_matrix = adj_matrix.at[i, connected_node].set(1)
      adj_matrix = adj_matrix.at[connected_node, i].set(1) # Undirected graph

  return adj_matrix

# --- activation and scan function ---
def forward(params):
    """
        params: w_in - input layer
                w_res = reservoir layer
                alpha = leaky integration param
    """
    w_in, w_res, alpha = params

    def inner(carry, x_t):
        h_prev = carry
        x_aug = jnp.concatenate([jnp.array([1.0]), x_t])  # add bias term
        x_tilde = jnp.tanh(w_in @ x_aug + w_res @ h_prev)
        h = ( 1 - alpha) * h_prev + alpha * x_tilde       # leaky integrate        
        return h, h
    return inner

# --- run reservoir ---
def reservoir(u, w_in, w_res, alpha=1):
    """
        u: inputs shape + 1 for bias
        w_in: input layer, shape (reservoir_dimension, inputs dim + 1)
        w_res: reservoir square matrix
        alpha: int leaky integration, less than one for slower dynamics
    """
    x0 = jnp.zeros(w_res.shape[0])
    forward_fn = forward((w_in, w_res, alpha))
    _, hidden_states = lax.scan(forward_fn, x0, u)
    return hidden_states 

# --- prediction function ---
def predict(u, w_in, w_res, alpha=1):
    h0 = jnp.zeros(100)
    forward_fn = forward((w_in, w_res, alpha))
    _, hidden_states = lax.scan(forward_fn, h0, u)
    h_aug = jnp.concatenate([hidden_states, jnp.ones((hidden_states.shape[0], 1))], 
                            axis=1)
    y_pred = h_aug @ w_out
    return y_pred

# --- func for sine wave ---
def sine_data(steps):
    t = jnp.linspace(0, 10, steps)
    return jnp.sin(t).reshape(-1, 1), t

# --- funcs for generating lorenz attractor ---
def f(state, t, rho, sigma, beta):
    x, y, z = state
    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

def odeint_rk4(f, y0, t, *args):
    def step(state, t):
        y_prev, t_prev = state
        h = t - t_prev
        k1 = h * f(y_prev, t_prev, *args)
        k2 = h * f(y_prev + k1/2., t_prev + h/2., *args)
        k3 = h * f(y_prev + k2/2., t_prev + h/2., *args)
        k4 = h * f(y_prev + k3, t + h, *args)
        y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y
    _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])
    return ys

def plot_3d_path(ax, ys, color, filename="fig.pdf"):
  x0, x1, x2 = ys.T
  ax.plot(x0, x1, x2, lw=0.5, color=color)
  plt.savefig(filename)
 
if __name__ == 'main':
   # --- could go with just a sine wav ---

  u = odeint_rk4(f, 
            jnp.array([5., 5., 5.]), 
            jnp.linspace(0, 10., 10_000), 
            28., 10., 8./3)

  fig = plt.figure(figsize=(6, 4), dpi=150)
  ax = fig.add_subplot(projection='3d')
  plot_3d_path(ax, u, 'blue', "lorenz.pdf");

  k = random.key(42) # Douglas Adams!
  k, k1, k2, k3, k4 = random.split(k, 5)

# --- set up layers --- #
  sparse_mask = (random.uniform(k1, (100, 100)) < 0.1).astype(jnp.float32)
  w_res_raw = random.normal(k3, (100, 100))
  w_res_sparse = w_res_raw * sparse_mask

  w_in = random.normal(k4, (100, 4))  # x, y, z of lorenz plus bias

# --- scale spectral radius --- #
  eigvals = jnp.linalg.eigvals(w_res_sparse)
  w_res = w_res_sparse * (0.9 / jnp.max(jnp.abs(eigvals)))

# --- run reservoir --- #
  hidden_states = reservoir(u, w_in, w_res)

# --- train readout ---
  h_aug = jnp.concatenate([hidden_states, jnp.ones((hidden_states.shape[0], 1))], axis=1)
  w_out = jnp.linalg.lstsq(h_aug, u, rcond=None)[0]

  # --- TODO ---
  # speed up the least squares with a cholesky decomp

  y_pred = predict(u, w_in, w_res)

  fig = plt.figure(figsize=(6, 4), dpi=150)
  ax = fig.add_subplot(projection='3d')
  plot_3d_path(ax, y_pred, 'violet', filename="reservoir_lorenz.pdf");

  print(f"### MSE: {((u - y_pred)**2).mean()} ###")
    
    

