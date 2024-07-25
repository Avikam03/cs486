![](assets/cover.png)

Stuff I can't remember / think is worth memorizing 🕺

## lec2
### Calculate Joint Probability Using the Chain Rule

$$
\begin{align*}
P(A_n, A_{n - 1}, \ldots, A_2, A_1) &= \prod_{i = 1}^n P(A_i \mid A_{i - 1} \land \dots \land  A_1) \\
&= P(A_n \mid A_{n-1}, A_{n-2}, \ldots, A_1) \cdot P(A_{n-1} \mid A_{n-2}, \ldots, A_1) \cdot \ldots \cdot P(A_2 \mid A_1) \cdot P(A_1) 
\end{align*}
$$

### Cool Bayes Rule trick

Don't need to know $P(Y)$ to compute $P(X \mid Y)$. $P(Y)$ is simply a normalization constant. We can calculate $P(X \mid Y)$ and $P(\neg X \mid Y)$ and then normalize them to sum to $1$

![](assets/lec2.1.png)

### Calculating Probabilities

A universal approach
1. To calculate a conditional probability, convert it into a fraction of two joint probabilities using the product rule in reverse.
2. To calculate a joint probability (not involving all the variables), write it as a summation of joint probabilities (involving all the variables) by introducing the other variables using the sum rule in reverse.
3. Calculate every joint probability (involving all the variables) using the chain rule.

## lec3

### Unconditional Independence

$X$ and $Y$ are unconditionally independent if
- $P(X \mid Y) = P(X)$ 
- $P(Y \mid X) = P(Y)$
- $P(X \land Y) = P(X) \cdot P(Y)$

need to make 4 comparisons

### Conditional Independence

$X$ and $Y$ are conditionally independent given $Z$ if
- $P(X \mid Y \land Z) = P(X \mid Y)$
- $P(Y \mid X \land Z) = P(Y \mid X)$
- $P(X \land Y \mid Z) = P(X \land Y)$

need to make 8 comparisons

**Note**: Independence does not imply conditional independence, and vice versa.

### Bayesian Networks

- Directed Acyclic Graph
- Each node is a random variable - can be continuous or discrete
- Represents conditional dependencies
- If an arrow is from $X$ to $Y$, then $X$ has a direct influence on $Y$
- **Markov Blanket**: a set of neighbouring variables that directly effect the variable $X$'s value. given this set of variables, $X$ is conditionally independent of all other variables.

Representing the joint distribution

$$P(X_n \land \dots \land X_1) = \prod_{i = 1}^n P(X_i \mid \text{Parents}(X_i))$$

### Three Key Structures

#### Structure 1

<img src="assets/lec3.1.png" width="400">

- Burglary and Watson are not independent
	- if Burglary is happening, Alarm is more likely to go off, meaning that Watson is more likely to call.
- Burglary and Watson are conditionally independent given Alarm

#### Structure 2

<img src="assets/lec3.2.png" width="250">

- Watson and Gibbon are not independent
	- If Watson is more likely to call, Alarm is more likely to go off, meaning that Gibbon is more likely to call.
- Watson and Gibbon are conditionally independent given Alarm

#### Structure 3

<img src="assets/lec3.3.png" width="250">

- Earthquake and Burglary are independent
- Earthquake and Burglary not conditionally independent given Alarm
	- Assume Alarm is going off. If Earthquake did cause it, less likely Burglary is also happening. If Burglary caused it, less likely Earthquake is also happening.


## lec4

### D-Separation

$E$ d-separates $X$ and $Y$ iff $E$ blocks all undirected paths between $X$ and $Y$.

if $E$ d-separates $X$ and $Y$, then $X$ and $Y$ are conditionally independent given $E$.

### Blocked Paths

There are a few scenarios to consider while checking for blocked undirected paths
#### Scenario 1

<img src="assets/lec4.1.png" width="400">

#### Scenario 2

<img src="assets/lec4.2.png" width="400">
#### Scenario 3

<img src="assets/lec4.3.png" width="400">

### Constructing Bayesian Networks

- For a joint probability distribution, there are many correct Bayesian networks.
- Given a Bayesian network A, a Bayesian network B is correct if and only if the following is true: If Bayesian network B requires two variables to satisfy an independence relationship, Bayesian network A must also require the two variables to satisfy the same independence relationship.
- We prefer a Bayesian network that requires fewer probabilities.

**Important**
- Bayesian network B could miss independence from Network A, but it cannot miss dependence.

#### Steps
- Order the variables $\{X_1, X_2, \dots, X_n\}$
- For each variable $X_i$ in order:
	- Pick the smallest subset $\text{Parents}(X_i)$ from $X_1, \dots, X_{i - 1}$ such that given $\text{Parents}(X_i)$, $X_i$ is independent from all variables $\{X_1, X_2, \dots, X_n\} - \text{Parents}(X_i)$
	- Create a link from each of $\text{Parents}(X_i)$ to $X_i$
	- Compute table for $P(X_i \mid \text{Parents}(X_i))$

### Important Notes about Bayesian Networks (from Alice Gao's videos)

- Not every link in the Bayesian network is representing a causal relationship
- For the original Bayesian network for the home scenario, it just so happens that the network was constructed in a way such that the causes were added before the corresponding effects, so all of the links represented causal relationship - however, this is not generally the case
- All of the links in bayesian networks can be reversed, so every link is representing some correlation, but NOT necessarily a causal relationship
- Changing the order of the variables while constructing bayesian networks can lead to having more/less links in the network. More links = bad = more complicated
- A general hand wavy rule that can be used to construct a compact Bayesian network is to pick a variable ordering that respects the causal relationship, so if there’s a causal relationship between nodes, always try add the causes before you add the effects.

Finding the most compact Bayesian Network is **NP-hard**!

### Causality vs Correlation

correlation does not imply causation :)

There may exist hidden confounding variables that effect both the supposed cause and effect.

**Example**: Experiments/Studies show a strong positive correlation between children reading skills and shoe size. Why are these two variables so correlated?

<img src="assets/lec4.4.png" width="300">

Here, Age is a the confounding variable. The hidden variable Age confounds the relationship between Shoe Size and Reading.

#### Intervention

To determine if there is a causal relationship, the concept of intervention is used.

We intervene in the system to manipulate one variable and observe the effect on another variable.

#### Average Treatment Effect (ATE)

ATE Measures the average effect of a treatment (in this case, shoe size) on an outcome (reading skills) across a population.

$$\text{ATE} = \sum_A p (R \mid S = 1, A) p(A) - \sum_A p(R \mid S = 0, A) p(A)$$

By subtracting these sums, we obtain the ATE, which tells us the average effect of changing shoe size on reading skills, accounting for the distribution of age.

If $\text{ATE} \approx 0$, then there is an indication that shoe size does not have a causal effect on reading skills. The observed correlation is explained by the confounding variable, age, which affects both shoe size and reading skills independently.

## lec5

### Supervised Learning

**No free lunch theorem**: In order to learn something useful, we have to make some assumptions — have an inductive bias.

How do we choose a hypothesis that generalizes well? One that predicts unseen data correctly?
- **Ockham’s razor**: prefer the simplest hypothesis consistent with the data
- **Cross-validation**: a more principled approach to choose a hypothesis

#### Biase Variance Tradeoff:

<img src="assets/lec5.1.png" width="500">

**Bias**: If I have infinite data, how well can I fit the data with my learned hypothesis

A hypothesis with a high bias 
- makes strong assumptions
- is too simplistic
- has few degrees of freedom
- does not fit the training data well

Problem with high bias: if can't even capture training data well, how can we possibly predict unseen data well?

**Variance**: How much does the learned hypothesis vary given different training data?

A hypothesis with high variance
- has a lot of degrees of freedom
- is very flexible
- fits the training data well

Problems with high variance: overfitting

<img src="assets/lec5.2.png" width="500">

#### Bias Variance Equation

Let $\hat{f} = \hat{f}(x; D)$

$$
\begin{align*}
\text{MSE} = E_{D, E} [(y - \hat{f})^2] = (\text{Bias}_D [\hat{f}])^2 + \text{Var}(\hat{f}) + \sigma^2
\end{align*}
$$

where

$$
\text{Bias}_D[\hat{f}] = E[\hat{f}] - f(x)
$$

$$
\text{Var}_D[\hat{f}] = E(E[\hat{f}] - \hat{f}(x))
$$

Thus, overall

$$\text{total error} = \text{bias} + \text{variance} + \text{irreducible noise}$$

![](assets/lec5.3.png)
**Source**: Wikipedia :)

#### Cross-validation

Used for finding a hypothesis with low bias and low variance.

Used when we don't have access to test data – uses part of the training data as a surrogate for test data (called validation data). Use validation data to choose hypothesis
##### Steps
- Break data into $k$ equally sized parts
- Train a learning algorithm on $k - 1$ parts (training set)
- Test on the remaining $1$ part (validation set)
- Do this $k$ times, each time testing on a different partition
- Calculate the average error on the $k$ validation sets

In the end, we can either
- choose one of the $k$ trained hypotheses as the final hypotheses
- train a new hypothesis on all data – using parameters selected by cross validation

#### Overfitting

<img src="assets/lec5.4.png" width="300">


### Unsupervised Learning

#### K-means algorithm

Randomly select $k$ data points as initial centroids
- Assign each data point to the closest centroid.
- Re-compute the centroid using the current cluster memberships.
- If convergence is not met, repeat

#### Independent Component Analysis (ICA)

ICA is a unsupervised learning technique used to separate mixed signals into their independent sources.

**Example**: Cocktail party problem: listening in on one person's speech in a noisy room.

Let's say we observe data $x_i(t)$. We can model it using hidden variables $s_i(t)$:

$$x_i(t) = \sum_{j = 1}^m a_{ij} s_j(t) \quad\quad i = 1, \dots, n$$

where we assume
- the observed data $x$ has $n$ components 
- there are $m$ independent sources

or as a matrix decomposition

$$X = AS$$

where
- $a_{ij}$, or the matrix $A$ is a constant parameter called the *mixing parameter*
- $S$, or $s_j(t)$ are the hidden random factors, also called *independent components* or *source signals*

We estimate both $A$ and $S$ by observing $X$

**Good Resource**: https://www.geeksforgeeks.org/ml-independent-component-analysis/

## lec6

### Activation Functions

1. **Step Function**: $g(x)=1$ if $x > 0$ and $g(x)=0$ if $x < 0$
	- Simple to use, but not differentiable

<img src="assets/lec6.2.png" width="500">

2. **Sigmoid Function**:  $g(x) = \dfrac{1}{1 + e^{-kx}}$
	- For very large and very small $x$, $g(x)$ is close to $1$ or $0$
	- Approximates the step function. As $k$ is increased, the sigmoid function becomes steeper
	- Differentiable
	- Computationally Expensive

<img src="assets/lec6.3.png" width="500">

*Vanishing Gradient Problem*: when $x$ is very large or very small, $g(x)$ responds little to changes in $x$. The network does not learn further or learns very slowly

3. **ReLU function**: $g(x) = \max(0, x)$
	- Computationally efficient - network converges quickly
	- Differentiable

<img src="assets/lec6.4.png" width="500">

*Dying ReLU problem*:  When inputs approach $0$ or are negative, the gradient becomes $0$ and the model can't learn anything

4. **Leaky ReLU**: $g(x) = \max(0, x) + k \cdot \min(0, x)$
	- Small positive slope k in the negative area. Enables learning for negative input values

<img src="assets/lec6.5.png" width="500">

### Convolutional Kernel

- **stride (s)**: number of pixels to move
- **padding (p)**: 
- **filter (f)**: size of the kernel

$$
\text{Output dimensions} = \Bigg(\frac{i - f + 2p}{s} + 1\Bigg), \Bigg( \frac{j - f + 2p}{s} + 1 \Bigg)
$$

$$
\text{Parameter Size} = \text{input channels} \times \text{kernel size} \times \text{output channels}
$$

**Pro tip**: To visualize output, imagine sliding a $f \times f$ square top left to bottom right. How many times can you do this?

#### Example #1
We are processing an image of $32 \times 32 \times 3$. We use a kernel of size $5 \times 5$ with an output channel of $8$. We slide the window with a stride of $1$. What is the parameter size? What is the output dimension?

Thus, we have the following info:
- $i = 32$, $j = 32$
- input channels $= 3$
- kernel size $= 5$
- $s = 1$
- output channels $= 8$

$$
\text{Output dimensions} = \Bigg(\frac{32 - 5}{1} + 1\Bigg), \Bigg( \frac{32 - 5}{1} + 1 \Bigg) = (28), (28)
$$

$$
\text{Parameter Size} = 3 \times 5 \times 5 \times 8 = 600
$$
#### Example #2
Stack another convolutional layer of size $3 \times 3$ with a depth of $16$. We slide the window with a stride of $2$. What is the output dimension?

Thus, we have the following info:
- $i = 28$, $j = 28$ (using the output dimensions of prev question as input dimensions)
- kernel size $= 3$
- $s = 2$

$$
\text{Output dimensions} = \Bigg(\frac{28 - 3}{2} + 1\Bigg), \Bigg( \frac{28 - 3}{2} + 1 \Bigg) = (13.5), (13.5) = (13, 13) \quad
$$

*round down since there is no padding*

Since the depth is 16, the total number of neurons is $13 \times 13 \times 16 = 2704$


### Recurrent Networks

- We use RNNs when patterns in our data change with time
- Feeds outputs back into the model's inputs.
- Can support short-term memory. For the given inputs, the behaviour of the network depends on its initial state, which may depend on previous inputs.

$h_t = f_W(h_{t - 1}, x_t)$
$y_t = f_Y(h_t)$

$h_t = \tanh(W_{hh} h_{t - 1} + W_{xh} x_t)$
$y_t = W_{hy} h_t$


<img src="assets/lec6.6.png" width="450">

#### Example: Modelling Language

##### Training Stage

<img src="assets/lec6.7.png" width="500">

`vocabulary V=[h,e,l,o]`
Thus, we pass input into our model character by character. We can one-hot encode each character. 
**Example**: `h` corresponds to `enc = [1, 0, 0, 0]`

We then feed it into the hidden layer, which produces an output.

**Example**: `h` produces the output `[1.0, 2.2, -3.0, 4.1`, which encodes to a prediction of `o`. This prediction is wrong, since the correct next letter is `e`. Since we're doing training, the loss would be computed and used to adjust the model.

##### Testing Stage

<img src="assets/lec6.8.png" width="500">

**Note**: Asked Prof. Wenhu about this in class. The numbers in this example/image don't seem to be accurate with the predictions :/ In general, we would predict the character with the max value after applying the softmax function


## lec07 

- amazing [yt vid](https://www.youtube.com/watch?v=NE88eqLngkg) for momentum, nestorv momentum, adagrad, rmsprop, adam

## lec11

### (More) Common Activation Functions

#### Softmax

Recall the sigmoid function
$$h(a) = \frac{1}{1 + e^{-a}}$$
The sigmoid function works only for 2 dimensions (binary classification). it takes a real value and outputs a value between 0 and 1.

The softmax function on the other hand looks like this:
$$h(a)_i = \dfrac{e^{a_i}}{\sum_j e^{a_j}}$$
The softmax function is a generalization of the sigmoid function to several dimensions (multi-class classification). it takes a vector of real valued inputs and transforms them into a probability distribution (all between 0 and 1).

#### Tanh (hyperbolic tangent)

$$h(a) = \dfrac{e^a - e^{-a}}{e^a + e^{-a}}$$

<img src="assets/lec11.1.png" width="500">

#### Gaussian

$$h(a) = e^{-0.5 \Bigg(\dfrac{a - \mu}{\sigma}\Bigg)^2}$$


## lec17

algorithms <3
### Search Algorithms

A search problem is defined by:
- A set of states
- An initial state
- Goal state(s)
- A Successor (neighbour) Function: How to go from one state to another
- (Optionally) A cost associated with each action

A solution to a search problem is a path going from the initial state to a goal state (optionally with the least cost).

```python
'''
Pseudocode for a generic search algorithm
	- graph search graph
	- s     start node
	- goal  function that returns true if reached goal state
'''
def search(graph, s, goal):
	frontier = {"s"}

	while frontier:
		cur = frontier.remove() # assume we have some way to do this
		if goal(cur):
			return cur
		for nbr in cur.neighbour: # obtain neighbours using successor function
			frontier.append(nbr)

	return NO_PATH_FOUND
```

### Depth First Search (DFS)

- Treat frontier like a stack (LIFO)
- Intuitively: Search one path to completion before trying another path. Backtrack to alternative if exhausted current path.
#### Properties

Useful Quantities:
- **branching factor (b)**: average number of children a node can have
- **maximum depth (m)**: of search tree
- **depth (d)**: of the shallowest goal node.

| Type                                                         | Complexity                       | Intuition                                                               |
| ------------------------------------------------------------ | -------------------------------- | ----------------------------------------------------------------------- |
| **Space Complexity**<br><br>(size of frontier in worst case) | $O(bm)$<br><br>linear in m       | remembers $m$ nodes in the current path, and $b$ siblings for each node |
| **Time Complexity**                                          | $O(b^m)$<br><br>exponential in m | visits the entire search tree in the worst case                         |

- DFS is NOT guaranteed to find a solution even if it exists. DFS will get stuck in an infinite path. This might happen because of cycles/loops in the graph, or simply because paths are infinitely long.
- DFS is NOT guaranteed to return the optimal solution if it terminates. This is because it doesn't consider costs.

#### When to use DFS
- when space is restricted
- when many solutions with long paths exist
#### When to not use DFS
- when there are infinite paths
- when solutions are shallow
- there are multiple paths to a node

### Breadth First Search (BFS)

- Treats frontier like a queue (FIFO)
- Intuitively: selects first encountered node with the least edges used so far
#### Properties

| Type                                                         | Complexity                       | Intuition                                                       |
| ------------------------------------------------------------ | -------------------------------- | --------------------------------------------------------------- |
| **Space Complexity**<br><br>(size of frontier in worst case) | $O(b^d)$<br><br>exponential in d | must visit the top $d$ levels                                   |
| **Time Complexity**                                          | $O(b^d)$<br><br>exponential in d | visits the entire search tree (up to level d) in the worst case |

- BFS is guaranteed to find a solution if it exists
- BFS is guaranteed to return an optimal solution if it terminates – assuming that all edges have the same cost. More generally, it is guaranteed to return the shallowest goal node.

#### When to use BFS
- when space isn't an issue
- want a solution with lowest number of edges (shallowest)
#### When to not use BFS
- when all solutions are deep in the tree
- problem is large and graph is dynamically generated 

### Iterative Deepening Search (IDS)

Combine the best parts of BFS and DFS to get IDS

best part of DFS
- needs less space: $O(bm)$
best part of BFS
- needs less runtime: $O(b^d)$
- guaranteed to find solution if it exists

#### How does it work?
For each depth limit, perform DFS until the limit is reached.

Note that we perform DFS from scratch each time, and don't retain any information from previous DFS runs for lesser depth limits.

**Intuition**: To me, this seems like BFS, but it's not. It seems like we're just expanding everything level by level. However, in actuality, it is only BFS in the sense that it only traverses till a certain depth at a time. This is why IDS inherits time complexity from BFS. However, the problem with BFS is that the frontier is too large. DFS on the other hand has a small frontier. By performing DFS at each depth, we can ensure that our frontier has the same maximum size as that of DFS.

#### Properties

| Type                                                         | Complexity                                        | Intuition                                    |
| ------------------------------------------------------------ | ------------------------------------------------- | -------------------------------------------- |
| **Space Complexity**<br><br>(size of frontier in worst case) | $O(bd)$<br><br>linear in d<br>(like DFS)          | guaranteed to terminate at depth $d$         |
| **Time Complexity**                                          | $O(b^d)$<br><br>exponential in d<br>(same as BFS) | visits all nodes up to level d in worst case |

- IDS is guaranteed to find a solution if it exists (Same as BFS)
- IDS is guaranteed to return an optimal solution if it terminates – assuming that all edges have the same cost. More generally, it is guaranteed to return the shallowest goal node (Same as BFS)

#### When to use IDS
- when space isn't an issue
- want a solution with lowest number of edges (shallowest)
#### When to not use IDS
- when all solutions are deep in the tree
- problem is large and graph is dynamically generated 

## lec18

### Heuristic Search

Instead of picking states randomly, and not knowing if a state is better than another (like how uninformed search algorithms operate), it would be much more efficient to have some sort of heuristic to estimate how close a state is to a goal. This can help us find the optimal solution faster!

#### Search Heuristic function
A search heuristic $h(n)$ is an estimate of the cost of the cheapest path from node n to a goal node.

Properties of good heuristics:
- problem-specific
- non-negative
- $h(n) = 0$ is $n$ is a goal node
- $h(n)$ must be easy to compute without search

#### Cost function
Suppose that we are executing a search algorithm and we have added a path ending at n to the frontier. $\text{cost}(n)$ is the actual cost of the path ending at n.

### Lowest Cost First Search (LCFS)

LCFS works by removing the path with the lowest cost $\text{cost}(n)$

**Fun fact**: this is Dijkstra's algorithm :)

| Property                     | Note                                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Space, Time Complexity**   | Exponential. <br>LCFS examines a lot of paths to ensure that it returns the optimal solution first.                                        |
| **Completeness, Optimality** | Yes, and yes under mild conditions<br>- The branching factor is finite<br>- The cost of every edge is bounded below by a positive constant |

### Greedy Best First Search (GBFS)

GBFS removes the path with the lowest heuristic value $h(n)$

| Property                   | Note                                                 |
| -------------------------- | ---------------------------------------------------- |
| **Space, Time Complexity** | Exponential                                          |
| **Complete**               | Not Complete. CBFS might get stuck in a cycle        |
| **Optimal**                | Not Optimal. GBFS may return an unoptimal path first |

### A* search

Removes the path with the lowest cost + heuristic value: $f(n) = \text{cost}(n) + h(n)$

| Property                     | Note                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| **Space, Time Complexity**   | Exponential.                                                 |
| **Completeness, Optimality** | Yes, and yes under mild conditions on the heuristic function |

#### A* is optimal
##### Theorem: Optimality of A*

A* is optimal iff $h(n)$ is admissible

##### Definition: Admissible Heuristic

A heuristic $h(n)$ is admissible if it never overestimates the cost of the cheapest path from node $n$ to the goal node.

##### Proof that A* is optimal

Assume we have many paths in the frontier such that $C^* < C^n$
$$(S \to G : C^{*}, \dots, S \to N : C^{n})$$
Let there be a path $S \to N \to G$ (not in the frontier) that has cost $C'$ such that $C' < C^*$

According to admissibility, $C^n < C' < C^*$.

However, this contradicts our assumption that $C^* < C^n$. Thus, A* is optimal!

##### A* is optimally efficient

- Among all optimal algorithms that start from the same start node and use the same heuristic, A* expands the fewest nodes.
- No algorithm with the same information can do better.
- A* expands the minimum number of nodes to find the optimal solution.

**Intuition for proof**: any algorithm that does not expand all nodes with $f(n) < C^*$ run the risk of missing the optimal solution. 

TODO: write the contradiction proof

### Designing an Admissible Heuristic

1. Define a relaxed problem: Simplify or drop one of the existing constraints in the problem.
2. Solve the relaxed problem without search
3. The cost of the optimal solution to the relaxed problem is an admissible heuristic for the original problem.

**Intuition**: The cost of the optimal solution for the easier problem should be lesser than the corresponding cost for the actual problem.
