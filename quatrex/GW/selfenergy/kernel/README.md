# Functions to Calculate the Self-Energies
## Problem Description
The input of Green's Functions and the screening is in the 2D format.

The self-energies are calculated in the following way on a discrete energy grid:

$$\Sigma^{\lessgtr}_{ij} \left(E\right) = 
i*\frac{dE}{2*\pi} \sum \limits_{E^{\prime}} 
G^{\lessgtr}_{ij}\left(E^{\prime}\right) W^{\lessgtr}_{ij}\left(E-E^{\prime}\right)$$


$$\Sigma^{r}_{ij} \left(E\right) = 
i*\frac{dE}{2*\pi} \sum \limits_{E^{\prime}} 
G^{r}_{ij}\left(E^{\prime}\right) W^{<}_{ij}\left(E-E^{\prime}\right) +
G^{>}_{ij}\left(E^{\prime}\right) W^{r}_{ij}\left(E-E^{\prime}\right)$$

The convolution theorem is used instead of the direct evaluation of the sum.
This is due to the lower complexity in the number of energy points.
The convolution is the same as the inverse Fourier transform of the product
of both Fourier transformations with the right padding.

In addition, in the previous step the energy grid for W/P 
got cut off to the same as G. Therefore, we need to adapt the formula above
to account for this error. 
The derivation is sketched out below.

## Side Notes
- It is not derived why the double counting of energy zero happens
- All the derivations are for the continuous case
- Only FFT implementation exists, since direct convolution calculations are not competitive, but would be nice to test against.

## Derivation

Again the formula for $\Sigma^{\lessgtr,r}$ are given as:

$$\Sigma^{\lessgtr}_{ij} \left(E\right) = \alpha \int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} G^{\lessgtr}_{ij} \left(E^{\prime}\right) W^{\lessgtr}_{ij} \left(E- E^{\prime}\right)$$

$$\Sigma^{r}_{ij} \left(E\right) = \alpha \int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} \left[ G^{r}_{ij} \left(E^{\prime}\right) W^{<}_{ij} \left(E- E^{\prime}\right) + G^{>}_{ij} \left(E^{\prime}\right) W^{r}_{ij} \left(E- E^{\prime}\right) \right]$$

Then without loss of generality, we assume that:

$$W^{\lessgtr,r,a}\left(E\right) = W^{\lessgtr,r}_{+}\left(E\right) + W^{\lessgtr}_{-}\left(E\right)$$

$$W^{\lessgtr,r,a}_{+}\left(E\right) \neq 0, \quad E \in \left[0, E_n\right]$$

$$W^{\lessgtr,r,a}_{-}\left(E\right) \neq 0, \quad E \in \left[-E_n,0\right]$$

$$G^{\lessgtr,r,a} \left(E\right) \neq 0, \quad E \in \left[0, E_n\right]$$

Therefore, the above integrals can be split up in the following way:

$$\Sigma^{\lessgtr}_{ij} \left(E\right) = \alpha \left\{\int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} G^{\lessgtr}_{ij} \left(E^{\prime}\right) W^{\lessgtr}_{ij,+} \left(E- E^{\prime}\right) + \int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} G^{\lessgtr}_{ij} \left(E^{\prime}\right) W^{\lessgtr}_{ij,-} \left(E- E^{\prime}\right)  \right\}$$

$$\Sigma^{r}_{ij} \left(E\right) = \alpha \left\{\int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} \left[ G^{r}_{ij} \left(E^{\prime}\right) W^{<}_{ij,+} \left(E- E^{\prime}\right) + G^{>}_{ij} \left(E^{\prime}\right) W^{r}_{ij,+} \left(E- E^{\prime}\right) \right] + \int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} \left[ G^{r}_{ij} \left(E^{\prime}\right) W^{<}_{ij,-} \left(E- E^{\prime}\right) + G^{>}_{ij} \left(E^{\prime}\right) W^{r}_{ij,-} \left(E- E^{\prime}\right) \right]\right\}$$

Where for the convolution theorem can be used for the positive part:

$$\int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} G^{\lessgtr}_{ij} \left(E^{\prime}\right) W^{\lessgtr}_{ij,+} \left(E- E^{\prime}\right) = \mathcal{F}\left(\mathcal{F}^{-1}\left(G^{\lessgtr}_{ij}\right) \mathcal{F}^{-1}\left(W^{\lessgtr}_{ij,+}\right)\right)$$

$$\int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} \left[ G^{r}_{ij} \left(E^{\prime}\right) W^{<}_{ij,+} \left(E- E^{\prime}\right) + G^{>}_{ij} \left(E^{\prime}\right) W^{r}_{ij,+} \left(E- E^{\prime}\right) \right] = \mathcal{F}\left(\mathcal{F}^{-1}\left(G^{r}_{ij}\left(E\right)\right) \mathcal{F}^{-1}\left(W^{<}_{ij,+}\left(E\right)\right)\right) + \mathcal{F}\left(\mathcal{F}^{-1}\left(G^{>}_{ij}\left(E\right)\right) \mathcal{F}^{-1}\left(W^{r}_{ij,+}\left(E\right)\right)\right)$$

We need identities for the negative part since it is cut off:

$$W^{r} \left(-E\right) = W^{a} \left(E\right)^{T}$$

$$W^{r} \left(E\right) = W^{a} \left(E\right)^{H}$$

As a consequence:

$$W^{r} \left(-E\right) = W^{r} \left(E\right)^{*}$$

Furthermore, from the formula of $^{\lessgtr} \left(E\right)$:

$$W^{\lessgtr} \left(E\right) = W^{r} \left(E\right) P^{\lessgtr} \left(E\right) W^{r} \left(E\right)^{H}$$

$$W^{\lessgtr} \left( -E\right) = W^{r} \left(-E\right) P^{\lessgtr} \left(-E\right) W^{r} \left(-E\right)^{H}$$

With the above identity for $W^{r} \left(-E\right)$:

$$W^{\lessgtr} \left( -E\right) = W^{r} \left(E\right)^{*} P^{\lessgtr} \left(-E\right) W^{r} \left(E\right)^{T}$$

Then we need identities for $P^{\lessgtr} \left(-E\right)$:

$$P^{\lessgtr} \left(-E\right) = -P^{\gtrless} \left(E\right)^{*}$$


Which does not lead directly to the right result, but from the definition of $P^{\lessgtr} \left(E\right)$:

$$P^{\lessgtr}_{ij} \left(E\right) = i\beta \int \,\text{d}E^{\prime} G^{\lessgtr}_{ij} \left(E^{\prime}\right) G^{\gtrless}_{ij} \left(E^{\prime} - E\right)$$

Evaluating the following:

$$\left(-P^{\lessgtr}_{ij} \left(E\right) \right)^{H} = \left(-P^{\lessgtr}_{ji} \left(E\right) \right)^{*}$$

$$=- \left(i\right)^{*}\beta \int \,\text{d}E^{\prime} G^{\lessgtr}_{ji} \left(E^{\prime}\right)^{*} G^{\gtrless}_{ji} \left(E^{\prime} - E\right)^{*}$$

With:

$$G^{\lessgtr} \left(E\right) = -G^{\lessgtr} \left(E\right)^{H} \Rightarrow G^{\lessgtr}_{ij} \left(E\right) = -G^{\lessgtr}_{ji} \left(E\right)^{*}$$

Leads to:

$$= i \beta \int \,\text{d}E^{\prime} G^{\lessgtr}_{ii} \left(E^{\prime}\right) G^{\gtrless}_{ij} \left(E^{\prime} - E\right)$$

$$\Rightarrow \left(-P^{\lessgtr}_{ij} \left(E\right) \right)^{H} = P^{\lessgtr}_{ij} \left(E\right)$$

$$\Rightarrow W^{\lessgtr} \left(-E\right) = W^{\gtrless} \left(E\right)^{T}$$

with:

$$\alpha = i \beta, \quad \beta \in \mathbb{R}$$

With the derived identities, the negative parts of the above integrals can be evaluated. First, the symmetry of the convolution to exchange the arguments is used, and a change of variables $E^{\prime} = - E^{\prime\prime}$ is made :

$$\int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} G^{\lessgtr}_{ij} \left(E- E^{\prime}\right) W^{\lessgtr}_{ij,-} \left(E^{\prime}\right) = \int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime\prime} G^{\lessgtr}_{ij} \left(E+ E^{\prime\prime}\right) W^{\lessgtr}_{ij,-} \left( -E^{\prime\prime}\right)$$

$$\int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime} \left[ G^{r}_{ij} \left(E-E^{\prime}\right) W^{<}_{ij,-} \left(E^{\prime}\right) + G^{>}_{ij} \left(E- E^{\prime}\right) W^{r}_{ij,-} \left(E^{\prime}\right) \right] = \int \limits_{-\infty}^{\infty}\,\text{d}E^{\prime\prime} \left[ G^{r}_{ij} \left(E + E^{\prime\prime}\right) W^{<}_{ij,-} \left(-E^{\prime\prime}\right) + G^{>}_{ij} \left(E+ E^{\prime\prime}\right) W^{r}_{ij,-} \left(-E^{\prime\prime}\right) \right]$$

Then the two identities $W^{\lessgtr} \left(-E\right) = W^{\gtrless} \left(E\right)^{T}$ and $W^{\lessgtr} \left( -E\right) = - W^{\gtrless} \left( E\right)^{*}$ are inserted. The following final expressions can be derived through close inspection of the terms and applying the non-empty domain to limit the integration:

$$\Sigma^{\lessgtr} \left(E\right) =  \alpha \left\{\mathcal{F}\left(\mathcal{F}^{-1}\left(G^{\lessgtr}_{ij}\left(E\right)\right) \mathcal{F}^{-1}\left(W^{\lessgtr}_{ij,+}\left(E\right)\right)\right) + \mathcal{F}\left(\mathcal{F}^{-1}\left(G^{\lessgtr}_{ij} \left(-E\right)\right) \mathcal{F}^{-1}\left(W^{\gtrless}_{ji,+}\left(E\right)\right)\right)\left(-E\right)  \right\}$$

$$\Sigma^{r} \left(E\right) = \alpha \left\{\mathcal{F}\left(\mathcal{F}^{-1}\left(G^{r}_{ij}\left(E\right)\right) \mathcal{F}^{-1}\left(W^{<}_{ij,+}\left(E\right)\right)\right) + \mathcal{F}\left(\mathcal{F}^{-1}\left(G^{>}_{ij}\left(E\right)\right) \mathcal{F}^{-1}\left(W^{r}_{ij,+}\left(E\right)\right)\right) + \mathcal{F}\left(\mathcal{F}^{-1}\left(G^{r}_{ij} \left(-E\right)\right) \mathcal{F}^{-1}\left(W^{<}_{ji,+}\left(E\right)\right)\right)\left(-E\right) + \mathcal{F}\left(\mathcal{F}^{-1}\left(G^{>}_{ij} \left(-E\right)\right) \mathcal{F}^{-1}\left(W^{r}_{ij,+}\left(E\right)^{*}\right)\right)\left(-E\right)  \right\}$$

### Notes:

- ${F}^{-1}\left(W^{r}_{ij,+}\left(E\right)^{*}\right)$ is the same as $\hat{W}^{r}_{ij,+}\left(-t\right)^{*}$

- When discretizing and replacing the integrals with sums, one should not count twice the $E^{\prime}=0$ element.

- todo how to derive that the "removal" is $\mathcal{F}^{-1}\left(G_{ij} \left(-E\right)\right)\left(t\right) W_{ij}\left(E=0\right)$ or $\mathcal{F}^{-1}\left(G_{ij} \left(-E\right)\right)\left(t\right) W_{ji}\left(E=0\right)$ depending on greater/lesser/retarded self-energy


