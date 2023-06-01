# 2D_DarcyProblem

We are interested in studying the steady-state solution of the 2D Darcy equation, which is a second-order linear elliptic partial differential equation (PDE) of the form:

$$ - \nabla \cdot ( \alpha(x,y) \nabla u(x,y) ) = s(x,y), \quad \text{in } \Omega = \left(0,1 \right)^2, $$

with homogeneous Dirichlet boundary conditions, i.e. 

$$ u(x,y) = 0 \quad \text{on } \partial \Omega = \partial \left(0,1 \right)^2.$$

Here, $\alpha \in L^{\infty} \left( (0,1)^2 ; \mathbb{R}_{+} \right)$ is the diffusion coefficient, and $s \in L^2\left( (0,1)^2 ; \mathbb{R}  \right)$ is the forcing function. We aim to find an approximate solution to problem \eqref{eq: Darcy problem} for different values of the diffusion coefficient, using three different methods: the Finite Element Method (FEM), Physics-Informed Neural Networks (PINNs), and Fourier Neural Operator. In the first two methods, we obtain an approximation of the solution by solving a single instance of the PDE for different values of $a$, while the Fourier Neural Operator learns an entire family of PDEs.

For the purpose of this study, we fix the forcing function to $f=1$ and select the diffusion coefficients $\alpha$ from a distribution of piecewise constant functions. In particular, we divide the domain into four equal subdomains

                                    -------------.-------------
                                   |  $\Omega_2$ |  $\Omega_3$ |
                                    -------------.-------------
                                   |  $\Omega_1$ |  $\Omega_4$ |
                                    -------------.-------------
                                   
and assume that $\alpha$ remains constant within each subdomain. Moreover, we enforce the constraint $0 < \alpha < 10$ to further refine the parameter space. 


-Package versions:

FEM --> Python: version 3.7.12 \\
        FEniCS: version 2019.1.0 \\
        Numpy: version 1.21.6 \\


PINNs -->

FNO -->

