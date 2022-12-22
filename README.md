# WDA-nepv

WDA-nepv[1] is a Bi-level Nonlinear Eigevector algorithm for Wasserstein Discriminant Analysis (WDA)[2].

## Background on Wasserstein Discriminant Analysis

### Firstly on LDA

Before the discussion on WDA, let us recall Linear Discriminant Analysis (LDA), another supervised linear dimensionality reduction problem similar to WDA:

Suppose we have labeled data vectors $\\{\mathbf{x}_i^c\\}\_{i=1}^{n_c}$ in $\mathbb{R}^d$ belonging to classes $c=1,2,\ldots,C$, where $n_c$ denotes the number of data vectors in class $c$. The goal of LDA is to derive a linear projection $\mathbf{P}\in\mathbb{R}^{d\times p}$ such that the dispersion of the projected data vectors of different classes are maximized and the dispersion of the projected data vectors of same classes are minimized. The dispersion of the projected data vectors of different classes is quantified as
$$\sum\_{c,c'>c}\sum\_{ij}\frac{1}{n_cn\_{c'}}\\|\mathbf{P}^T(\mathbf{x}_i^c-\mathbf{x}_j^{c'})\\|_2^2$$
and the dispersion of the projected data vectors of same classes is quantified as
$$\sum\_{c}\sum\_{ij}\frac{1}{n_c^2}\\|\mathbf{P}^T(\mathbf{x}_i^c-\mathbf{x}_j^{c})\\|_2^2$$

Then, by rewriting the dispersions as trace operators, LDA is defined as the following optimization
$$\max_{\mathbf{P}^T\mathbf{P}=I_p}\frac{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_b\mathbf{P})}{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_w\mathbf{P})}$$

where $\mathbf{C}_b$ and $\mathbf{C}_w$ are the between and within cross-covariance matrices

$$\mathbf{C}_b = \sum\_{c,c'>c}\sum\_{ij}\frac{1}{n_cn\_{c'}}(\mathbf{x}_i^c-\mathbf{x}_j^{c'})(\mathbf{x}_i^c-\mathbf{x}_j^{c'})^T$$

$$\mathbf{C}_w = \sum\_{c}\sum\_{ij}\frac{1}{n_c^2}(\mathbf{x}_i^c-\mathbf{x}_j^{c})(\mathbf{x}_i^c-\mathbf{x}_j^{c})^T$$

### Now on WDA

WDA shares the same goal as LDA: derive a linear projection $\mathbf{P}$ such that the dispersion of the projected data vectors of different classes are maximized and the dispersion of the projected data vectors of same classes are minimized.

The main difference between WDA and LDA is that

```
WDA can dynamically control global and local relations of the data points
```

1. Unlike LDA that treats each data dispersion equally, WDA weighs each data dispersion differently. Specifically, in LDA, while the dispersion $\\|\mathbf{P}^T(\mathbf{x}_i^c-\mathbf{x}_j^{c'})\\|_2^2$ is weighted by a constant $\frac{1}{n_cn\_{c'}}$, in WDA, the dispersion is weighted by ${T}\_{ij}^{c,c'}(\mathbf{P})$.
2. The weights ${T}\_{ij}^{c,c'}(\mathbf{P})$ are determined as the components of the optimal transport matrix $\mathbf{T}^{c,c'}(\mathbf{P})\in\mathbb{R}^{n_c\times n\_{c'}}$. In short, the optimal transport matrix is computed using the regularized Wasserstein distance[cuturi] as the distance metric for the underlying empirical measure of the data vectors. See Example 1 for a detailed explanation on the optimal transport matrix.
3. The advantage of WDA is that the hyperparameter $\lambda$ in the optimal transport matrix $\mathbf{T}^{c,c'}(\mathbf{P})$ controls the weights ${T}\_{ij}^{c,c'}(\mathbf{P})$. In particular, small $\lambda$ puts emphasis on global relation of the data vectors while larger $\lambda$ favors locality structure of the data manifold. In fact, when $\lambda=0$ WDA is precisely LDA.

In summary, the dispersion of the projected data vectors of different classes is quantified as
$$\sum\_{c,c'>c}\sum\_{ij}{T}\_{ij}^{c,c'}(\mathbf{P})\\|\mathbf{P}^T(\mathbf{x}_i^c-\mathbf{x}_j^{c'})\\|_2^2$$
and the dispersion of the projected data vectors of same classes is quantified as
$$\sum\_{c}\sum\_{ij}{T}\_{ij}^{c,c}(\mathbf{P})\\|\mathbf{P}^T(\mathbf{x}_i^c-\mathbf{x}_j^{c})\\|_2^2$$

Then, WDA is defined as the following optimization
$$\max_{\mathbf{P}^T\mathbf{P}=I_p}\frac{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_b(\mathbf{P})\mathbf{P})}{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_w(\mathbf{P})\mathbf{P})}$$

where $\mathbf{C}_b(\mathbf{P})$ and $\mathbf{C}_w(\mathbf{P})$ are the between and within cross-covariance matrices

$$\mathbf{C}_b(\mathbf{P}) = \sum\_{c,c'>c}\sum\_{ij}{T}\_{ij}^{c,c'}(\mathbf{P})(\mathbf{x}_i^c-\mathbf{x}_j^{c'})(\mathbf{x}_i^c-\mathbf{x}_j^{c'})^T$$

$$\mathbf{C}_w(\mathbf{P}) = \sum\_{c}\sum\_{ij}{T}\_{ij}^{c,c}(\mathbf{P})(\mathbf{x}_i^c-\mathbf{x}_j^{c})(\mathbf{x}_i^c-\mathbf{x}_j^{c})^T$$


## Discussion on algorithms

### Computing the optimal transport matrix

The problem of computing the optimal transport matrix can be formulated as a matrix balancing problem:

For the matrix $\mathbf{K}^{c,c'}:=\Big([e^{-\lambda\\|\mathbf{P}^T\mathbf{x}_i^c-\mathbf{P}^T\mathbf{x}_j^{c'}\\|_2^2}]\_{ij} \Big) \in\mathbb{R}^{n_c\times n\_{c'}}$, compute the vectors $\mathbf{u}\in\mathbb{R}\_{+}^{n_c}$ and $\mathbf{v}\in\mathbb{R}\_{+}^{n\_{c'}}$ such that
$$\mathbf{T}^{c,c'}(\mathbf{P})=\mathcal{D}(\mathbf{u})\mathbf{K}\mathcal{D}(\mathbf{v})$$
where $\mathcal{D}(\cdot)$ denotes a diagonal matrix with vector $\cdot$ as the diagonal.

Such matrix balancing problem can be solved by Sinkhorn-Knopp (SK) algorithm[2] or its accelerated variant (Acc_SK) based on a vector-dependent nonlinear eigenvalue problem (NEPv)[3].

Instead of SK, WDA-nepv uses Acc_SK to compute the optimal transport matrices. In Example 1, we show that SK algorithm is subject to slow convergence or even non-convergence while Acc_SK converges is just a few iterations.

### Other existing WDA algorithms

1. WDA-gd[3] uses the steepest descent method from the manifold optimization package Manopt[33] to solve WDA. The optimal transport matrices are computed by SK algorithm and their derivatives by automatic differentiation. As such, there are two downsides of this approach:
* The optimal transport matrices may be inaccurate, as illustrated in Example 1.
* Computing the derivatives are quite expensive - the cost grows quadratically in number of data vectors and in the dimension size.

2. WDA-eig[33] proposes a surrogate ratio-trace model
$$\max_{\mathbf{P}^T\mathbf{P}=I_p}\mbox{Tr}\bigg((\mathbf{P}^T\mathbf{C}_b(\mathbf{P})\mathbf{P})(\mathbf{P}^T\mathbf{C}_w(\mathbf{P})\mathbf{P})^{-1} \bigg)$$
to approximate WDA. Then, the following NEPv is further proposed as a solution of the surrogate model
$$\mathbf{C}_b(\mathbf{P})\mathbf{P}=\mathbf{C}_w(\mathbf{P})\mathbf{P}\Lambda$$
There are major flaws of this approach:
* WDA-eig also computes the optimal transport matrices by SK algorithm, thus they may be inaccurate.
* The surrogate ratio-trace model is a poor approximation to WDA.
* There is no theoretical result on the equivalence of the proposed NEPv and the surrogate model.

### WDA-nepv

WDA-nepv solves WDA by a simple iterative scheme:

$$\mathbf{P}\_{k+1} = \mbox{argmax}\_{\mathbf{P}^T\mathbf{P}=I_p}
\frac{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_b(\mathbf{P}_k)\mathbf{P})}
{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_w(\mathbf{P}_k)\mathbf{P})}$$



WDA-nepv is a bi-level nonlinear eigenvector (NEPv) algorithm:

* A NEPv for computing the optimal transport matrices, using the acclerated Sinkhorn-Knopp (Acc-SK) algorithm [3].
* A NEPv for computing the trace ratio problems.

Both NEPvs are solved efficiently by the self-consistent field (SCF) iteration.

In essence, WDAnepv updates the projection matrix by

$$\mathbf{P}\_{k+1} = \mbox{argmax}\_{\mathbf{P}^T\mathbf{P}=I_p}
\frac{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_b(\mathbf{P}_k)\mathbf{P})}
{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_w(\mathbf{P}_k)\mathbf{P})}$$

In contrast to the algorithm in [1], WDA-nepv is derivative-free, thus forgoes heavy costs of computing the derivatives.



# References

[1] 

[1] R ́emi Flamary, Marco Cuturi, Nicolas Courty, and Alain Rakotomamonjy. Wasserstein dis
criminant analysis. Machine Learning, 107(12):1923–1945, 2018.

[2] Marco Cuturi. Sinkhorn distances: lightspeed computation of optimal transport. In NIPS,
volume 2, page 4, 2013.

[3] A Aristodemo and L Gemignani. Accelerating the sinkhorn–knopp iteration by arnoldi-type
methods. Calcolo, 57(1):1–17, 2020.
