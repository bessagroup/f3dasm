from fidelity_augmentors import Warp, Scale, Noise_interpolator
import f3dasm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

dimensionality = 1
number_of_samples = 1000
seed = 123

noisy_data = 1
biased_data = 0

bounds = np.tile([0.0, 1.0], (dimensionality, 1))

fun_class_list = [fun for fun in f3dasm.functions.get_functions(d=1, randomized_term=False) if fun in f3dasm.functions.get_functions(d=500)]
# fun: f3dasm.functions.PyBenchFunction = fun_class_list[2](scale_bounds=bounds, dimensionality=dimensionality, offset=False)
# print(fun.name)

def scale(input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
    for dim in range(dimensionality):
        output_row *= 1 + np.sin(20 * np.pi * input_row[:, dim][:, None]) / 2
    return output_row

def reverse_scale(input_row: np.ndarray, output_row: np.ndarray) -> np.ndarray:
    output_row /= 1 + np.sin(5 * np.pi * input_row) / 2
    return output_row

a = 3.
warp_augmentor: f3dasm.functions.Augmentor = Warp(warp=lambda x: x ** a, reverse_warp=lambda y: y ** (1 / a))
scale_augmentor: f3dasm.functions.Augmentor = Scale(scale=scale, reverse_scale=reverse_scale)

aug_coeff = 0.5
noise_interpolator: f3dasm.functions.Augmentor = Noise_interpolator(noise_var=1., aug_coeff=aug_coeff)

design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=dimensionality)
sampler: f3dasm.Sampler = f3dasm.sampling.SobolSequence(design=design, seed=seed)

# np.random.seed(None)
i = np.random.randint(low=0, high=len(fun_class_list) - 1)
for fun_class in fun_class_list[i:i + 1]:

    # fun = f3dasm.functions.Periodic(scale_bounds=bounds, dimensionality=dimensionality, offset=False)
    fun = fun_class(scale_bounds=bounds, dimensionality=dimensionality, offset=False)

    ## Generate training samples with the sampler
    train_data = sampler.get_samples(numsamples=number_of_samples)
    fun(train_data)

    ## Extracting data for warping and scaling
    train_x = train_data.get_input_data().values.copy()
    train_y = train_data.get_output_data().values.copy()

    train_y_augmented = noise_interpolator.augment(train_y)

    # train_y_biased = scale_augmentor.augment(
    #         input_row=warp_augmentor.augment(
    #             input_row=train_x.copy()
    #         ), 
    #     output_row=fun(
    #         warp_augmentor.augment(
    #                 input_row=train_x.copy()
    #             )
    #         )
    #     )
    
    # scaler = StandardScaler()
    # scaler.fit(train_y)

    # train_y_scaled = scaler.transform(train_y).flatten()

    # noise = np.random.randn(*train_y_scaled.shape)
    # train_y_scaled += noise

    # train_y_noisy = scaler.inverse_transform(train_y_scaled[:, None])
    # noise = scaler.inverse_transform(noise[:, None])

    # if noisy_data:
    #     train_y_combi = train_y_noisy.flatten()

    # else:
    # train_y_combi = c * train_y.flatten() \
    #     + (1 - noisy_data / 2) * biased_data * (1 - c) * train_y_biased.flatten() \
    #     + (1 - biased_data / 2) * noisy_data * (1 - c) * noise.flatten()
        
    if dimensionality == 1:
        sort_indices = train_x.argsort(axis=0)
        train_x = train_x[sort_indices].squeeze(axis=-1)
        train_y = train_y[sort_indices].squeeze(axis=-1)
        train_y_augmented = train_y_augmented[sort_indices].squeeze(axis=-1)

    r2 = r2_score(y_true=train_y.flatten(), y_pred=train_y_augmented.flatten())
    cc = np.corrcoef(train_y.flatten(), train_y_augmented.flatten())[0, 1]

    if np.abs(cc) > 0.:
        
        print()
        print(fun.name)
        print('corr:', cc)
        print('r2:', r2)
        
        if dimensionality == 1:
            plt.figure(fun.name)
            plt.plot(train_x, train_y)
            plt.plot(train_x, train_y_augmented, '.')
            plt.legend(['Original', 'Augmented'])


plt.show()