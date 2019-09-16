%%
%{
This function can be used to generate a DoE for the loading space.
Inputs:
        dim: (scalar) Dimension of the input space (number of strain components)
        N: (scalar) Number of DoE samples (sites). The bare minimum is 10*dim.
        LowerBounds: (1 x dim vector) The lower bounds of each DoE site.
        UpperBounds: (1 x dim vector) The upper bounds of each DoE site.
Output:
        DoE: The desired DoE

example use for E11, E22, and E12: 
            DoE = DoE_Generator_For_2DStrain(3, 100, [-0.2, -0.2, -0.5], [2, 2, 2.2]);


Written by Ramin Bostanabad 02/27/2016, IDEAL
bostanabad@u.northwestern.edu
%}
function DoE = generate_DoE(dim, N, LowerBounds, UpperBounds)
    if ~all(size(LowerBounds) == [1, dim]) || ~all(size(UpperBounds) == [1, dim])
        error('Lower/upper bounds are  not specified correctly!.');
    end
    if ~all(UpperBounds > LowerBounds)
        error('The upper bounds should be all larger than the corresponding lower bounds!.');
    end
    DoE = sobolset(dim,'Skip',1e4);
    DoE = scramble(DoE,'MatousekAffineOwen');
    DoE = net(DoE, N).*repmat(UpperBounds - LowerBounds, N, 1) + repmat(LowerBounds, N, 1);
end
