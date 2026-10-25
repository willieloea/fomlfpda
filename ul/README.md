# Unsupervised Learning
(Ch. 4 of Computational Intelligence)

## 4.1 Background

## 4.4 Learning Vector Quantizer - I (LVQ)

**Goal of clustering algorithms:** Construct clusters of similar input vectors, where similarity is usually measured in terms of Euclidean distance.

An LVQ-I network consists of a set of input units, $\underline{z}$, a set of output units, $\underline{o}$, and a weight matrix between $\underline{z}$ and $\underline{o}$:

$$U_{K \times I} = \begin{bmatrix} u_{11} & u_{12} & \dots & u_{1I} \\ u_{21} & u_{22} & \dots & u_{2I} \\ \vdots & \vdots & \ddots & \vdots \\ u_{K1} & u_{K2} & \dots & u_{KI} \end{bmatrix}$$

The input units are the values of the descriptive features of an instance, and the output units are used to indicate different clusters.

```
  ↑ o_1          ↑ o_2    ...     ↑ o_K       (Output units / clusters)
  ▲  ▲  ▲        ▲  ▲  ▲          ▲  ▲  ▲
  │  └──┼────────┼──┼──┼──────────┼──┼──┐
  │ u_11│        │  │  │          │  │  │ u_KI
  │     │        │  │  │          │  │  │
  ○─────┴────────┘  ○──┴──────────┘  ○──┘     (Input units / features)
 z_1               z_2       ...    z_I
```
*(Each input unit $z_i$ for $i \in [1..I]$ is fully connected to each output unit $o_k$ for $k \in [1..K]$ via weight $u_{ki}$.)*

There are two ways the weight matrix can be initialized:
1. Sample $u_{ki}$ from a uniform distribution in the range of feature $i$.
2. Use the first $K$ training instances to set the weights, where $u_{ik} = z_{ik}$, where $i \in [1..I], \, k \in [1..K]$.

Each output unit's incoming weights $\underline{u}_k = \langle u_{k1}, u_{k2}, \dots, u_{ki} \rangle$ represent a **"prototype" vector**, which is a point used to represent a cluster.

### Training Algorithm

Training works by presenting input instances and moving the winning prototype towards each one:

1. Initialize the $K$ prototype vectors and choose a learning rate $\eta$.
2. Present an input $\underline{x}$, and find the closest prototype:

   $$\text{(1)} \quad k^* = \arg\min_k \|\underline{x} - \underline{u}_k\|_2 = \arg\min_k \sqrt{\sum_{i=1}^{I} (x_i - u_{ki})^2}$$

3. Move the winning prototype toward the input:

   $$\text{(2)} \quad \underline{u}_{k^*} \leftarrow \underline{u}_{k^*} + \eta \cdot (\underline{x} - \underline{u}_{k^*})$$

4. Repeat over the training data for multiple epochs, decreasing the learning rate over time.
5. Stop when a certain number of epochs are reached, or when prototype changes become sufficiently small.

### Conscience Mechanism

Poor initialization can lead to poor clustering. One prototype could keep winning, while another never does. To prevent this, you can use a **"conscience" mechanism** to determine the best prototype for an input instance:

$$\text{(3)} \quad \text{score}_k = \|\underline{x} - \underline{u}_k\|_2 + \lambda \left(f_k - \frac{1}{K}\right)$$

where:
- $f_k$ is prototype $k$'s estimated winning frequency.
- $\frac{1}{K}$ is an equal share of wins.
- $\lambda$ controls the penalty.

The prototype $\underline{u}_k$ with the lowest score is selected (Equation (3) replaces Equation (1)).

## 4.5 Self-Organizing Feature Maps (SOM)

A SOM is a projection of an $I$-dimensional feature space onto a two-dimensional grid, which should reflect the topology of the feature space. So SOMs can reduce dimensionality while retaining topological information.

### 4.5.1 Stochastic Training Rule

SOMs are mappings from an $I$-dimensional input space to a two-dimensional grid of neurons.

For a rectangular map with $K$ rows and $J$ columns, neuron $(k, j)$ has:
- a fixed grid position $r_{k,j} = (k, j)$
- a learned prototype vector $\underline{w}_{kj} \in \mathbb{R}^I$

```
            ( r_kj )
            ↗   ↑   ↖
     w_kj1 /  w_kj2  \ w_kjI
          /     │     \
       (x_1)  (x_2) ... (x_I)
```

So a SOM is essentially a $K \times J$ grid of weights, where each weight is an $I$-dimensional vector, $\underline{w}_{kj}$:

$$W \subseteq \mathbb{R}^{K \times J \times I}$$

To map an input onto the SOM, we find its closest prototype, the **best matching unit (BMU)**:

$$(k^*, j^*) = \arg\min_{k,j} \|\underline{x} - \underline{w}_{kj}\|_2$$

so $\underline{x} \rightarrow r_{k^*j^*}$.

When an input arrives, the winning prototype and its nearby neuron neighbours are moved:

$$\underline{w}_{kj} \leftarrow \underline{w}_{kj} + \eta \, h_{kj, k^*j^*} (\underline{x} - \underline{w}_{kj})$$

where:
- $\eta$ is the learning rate.
- $h_{kj, k^*j^*}$ determines how strongly neuron $(k, j)$ is updated, based on the grid distance between $(k, j)$ and $(k^*, j^*)$: $\|\underline{r}_{kj} - \underline{r}_{k^*j^*}\|_2$.

The function \(h\) controls **how strongly each neuron learns from an input, based on its grid distance from the winner**.

A common choice is a **Gaussian neighbourhood function**:

$$
\boxed{
h_{kj,k^*j^*}(t)
=
\exp\left(
-\frac{\|\mathbf{r}_{kj}-\mathbf{r}_{k^*j^*}\|_2^2}
{2\sigma(t)^2}
\right)
}
$$

Here:

- \(\mathbf{r}_{kj}\) is the neuron’s grid position.
- \(\mathbf{r}_{k^*j^*}\) is the winner’s grid position.
- \(\sigma(t)>0\) controls the neighbourhood’s width and decreases during training.

For a rectangular grid, the squared distance is simply

$$
\|\mathbf{r}_{kj}-\mathbf{r}_{k^*j^*}\|_2^2
=(k-k^*)^2+(j-j^*)^2.
$$

With \(\sigma=1\), the influence decreases like this:

| Grid distance from winner | \(h\) |
|---|---:|
| \(0\): winner itself | \(1\) |
| \(1\): adjacent neuron | \(0.607\) |
| \(2\) | \(0.135\) |
| \(3\) | \(0.011\) |

In the update

$$
\mathbf{w}_{kj}\leftarrow\mathbf{w}_{kj}
+\eta(t)\,h_{kj,k^*j^*}(t)
(\mathbf{x}-\mathbf{w}_{kj}),
$$

**\(\eta h\) is the effective learning rate for that neuron**. For example, if \(\eta=0.2\) and \(h=0.607\), the prototype moves about \(12.1\%\) of the way toward the input.

There is no single mandatory equation for \(h\). A simpler alternative uses a hard radius \(R(t)\):

$$
h_{kj,k^*j^*}(t)=
\begin{cases}
1,&\text{if grid distance}\le R(t),\\
0,&\text{otherwise}.
\end{cases}
$$

The Gaussian gives a gradual decline in influence; the hard-radius version updates every neuron inside the neighbourhood equally.
