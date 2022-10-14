# WDA-nepv

WDA-nepv, a Bi-level Nonlinear Eigevector algorithm for Wasserstein Discriminant Analysis.

Wasserstein Discriminant Analysis (WDA) [1] is a supervised linear dimensionality reduction problem:
$$\max_{\mathbf{P}^T\mathbf{P}=I_p}\frac{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_b(\mathbf{P})\mathbf{P})}{\mbox{Tr}(\mathbf{P}^T\mathbf{C}_w(\mathbf{P})\mathbf{P})}$$
where $\mathbf{C}_b(\mathbf{P})$ and $\mathbf{C}_w(\mathbf{P})$ are defined as the between and within cross-covariance matrices

$$\mathbf{C}_b(\mathbf{P}_k) = \sum\_{c,c'>c}\sum\_{ij}{T}\_{ij}^{c,c'}(\mathbf{P}_k)(\mathbf{x}_i^c-\mathbf{x}_j^{c'})(\mathbf{x}_i^c-\mathbf{x}_j^{c'})^T$$

$$\mathbf{C}_w(\mathbf{P}_k) = \sum\_{c}\sum\_{ij}{T}\_{ij}^{c,c}(\mathbf{P}_k)(\mathbf{x}_i^c-\mathbf{x}_j^{c})(\mathbf{x}_i^c-\mathbf{x}_j^{c})^T$$

The matrices $\mathbf{T}^{c,c'}$ and $\mathbf{T}^{c,c}$ are optimal transport matrices and can be computed by a matrix balancing algorithm, such as Sinkhorn-Knopp algorithm [2] or its accelerated variant [3].

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

[1] R ́emi Flamary, Marco Cuturi, Nicolas Courty, and Alain Rakotomamonjy. Wasserstein dis
criminant analysis. Machine Learning, 107(12):1923–1945, 2018.

[2] Marco Cuturi. Sinkhorn distances: lightspeed computation of optimal transport. In NIPS,
volume 2, page 4, 2013.

[3] A Aristodemo and L Gemignani. Accelerating the sinkhorn–knopp iteration by arnoldi-type
methods. Calcolo, 57(1):1–17, 2020.
