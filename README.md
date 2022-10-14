# WDAnepv

WDAnepv, a Bi-level Nonlinear Eigevector algorithm for Wasserstein Discriminant Analysis

Wasserstein Discriminant Analysis (WDA) is a supervised linear dimensionality reduction problem:
$$\max_{\mathbf{P}^T\mathbf{P}=I_p}\frac{\mbox{tr}(\mathbf{P}^T\mathbf{C}_b(\mathbf{P})\mathbf{P})}{\mbox{tr}(\mathbf{P}^T\mathbf{C}_w(\mathbf{P})\mathbf{P})}$$
where $\mathbf{C}_b(\mathbf{P})$ and $\mathbf{C}_w(\mathbf{P})$
are defined as the between and within cross-covariance matrices
$$\mathbf{C}_b(\mathbf{P}_k) = \sum_{c,c'>c}  \sum_{ij}{T}_{ij}^{c,c'}(\mathbf{P}_k)(\mathbf{x}_i^c-\mathbf{x}_j^{c'})(\mathbf{x}_i^c-\mathbf{x}_j^{c'})^T$$
$$\mathbf{C}_w(\mathbf{P}_k) = \sum_c  \sum_{ij}{T}_{ij}^{c,c}(\mathbf{P}_k)(\mathbf{x}_i^c-\mathbf{x}_j^{c})(\mathbf{x}_i^c-\mathbf{x}_j^{c})^T $$
The matrices $\mathbf{T}^{c,c'}$ and $\mathbf{T}^{c,c}$ are transport matrices 
