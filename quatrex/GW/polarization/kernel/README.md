# Functions to Calculate the Polarization. 
## Problem Description
The polarization is calculated in the following way on a discrete energy grid:

$$P^{\lessgtr}_{ij}\left(E^{\prime}\right) = -2i\frac{dE}{2 \pi} \sum \limits_{E} G^{\lessgtr}_{ij}\left(E\right) G^{\gtrless}_{ji}\left(E-E^{\prime}\right)$$
$$P^{r}_{ij}\left(E^{\prime}\right) = 
-2i\frac{dE}{2 \pi} \sum \limits_{E} G^{<}_{ij}\left(E\right) 
G^{a}_{ji}\left(E-E^{\prime}\right)+
G^{r}_{ij}\left(E\right) G^{<}_{ji}\left(E-E^{\prime}\right)$$

There are two main ways to evaluate the above sums:
Either directly evaluating the sum or FFT transforming with the convolution/correlation theorem.


## FFT Solution Idea

$$P^{r} = \alpha \left( G^{r}_{ij}(t) G^{<}_{ji}(-t) +
G^{<}_{ij}(t) \left(G^{r}_{ij}(t)\right)^{\prime} \right)$$

- FFT with a zero padding the number of energy points to comply with the convolution theorem.

- Reverse and transpose the second input due to the fact that $F(f^*(E))$ = $g^*(-t)$, where $F(f(E)) = g(t)$

- Then elementwise multiply the two inputs and elementwise multiply with a pre-factor in the time domain.
The pre-factor is given in the following way: $-2*i*dE/(2*pi)$ with an additional factor two due to spin

- Then IFFT

- Cut off points to get the polarization on the same energy grid as the Green's Function


## Side Notes
- In the time domain the following identity should hold:

$$P^{>}_{ij}\left(E\right) = -P^{<}_{ij}\left(-E\right)^{*}$$

The lesser or greater polarization can be calculated from this identity.
Thus, one can save a reversal/transpose operation.

- The elementwise multiplication with the pre-factor can be switched with 
applying the IFFT/cutting of elements

- The transposing and reversal could be switched with the FFT, 
but this would be less efficient since more FFT calls would be needed.

- GPU  implementation mirrors CPU one

- MPI implementation needs the transposed arrays as arguments 
since it is not local per rank possible
as transposing is a global operation

- Calculating the convolution directly is straightforward and needs no comments
- The convolution is slower than FFT
- Convolution time complexity: $\mathcal{O}\left(NE^2\right)$ where $NE$ is the number of energy points
- FFT time complexity: $\mathcal{O}\left(NE \log\left(NE\right)\right)$



