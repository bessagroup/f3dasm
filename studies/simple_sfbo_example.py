from matplotlib import pyplot as plt

from f3dasm import run_optimization, make_nd_continuous_design
from f3dasm.functions import pybenchfunction
from f3dasm.optimization.bayesianoptimization_torch import BayesianOptimizationTorch
from f3dasm.sampling import SobolSequence


def main():
    dim = 1
    fun = pybenchfunction.Schwefel(dimensionality=dim)
    parameter_DesignSpace = make_nd_continuous_design(
        bounds=fun.input_domain.astype(float),
        dimensionality=dim,
    )
    SobolSampler = SobolSequence(design=parameter_DesignSpace)
    samples = SobolSampler.get_samples(numsamples=8)
    samples.add_output(output=fun(samples))
    optimizer = BayesianOptimizationTorch(data=samples)
    optimizer.init_parameters()
    res = run_optimization(
        optimizer=optimizer,
        function=fun,
        sampler=SobolSampler,
        iterations=40,
        seed=123,
        number_of_samples=10,
    )

    # plot_x = np.linspace(fun.input_domain[0, 0], fun.input_domain[0, 1], 500)[:, None]
    # plt.plot(plot_x, fun(plot_x))
    # plt.scatter(res.data['input'], res.data['output'])
    # plt.show()


if __name__ == "__main__":
    main()
