import f3dasm
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from f3dasm.functions.fidelity_augmentors import NoiseInterpolator

def train_data_single_fidelity(config, sampler, fun):
    ## Generate training samples with the sampler
    train_data: f3dasm.ExperimentData = sampler.get_samples(numsamples=config.sampler.number_of_samples)
    fun(train_data)

    ## Extracting output data for scaling
    train_y = torch.tensor(train_data.get_output_data().values)

    ## Scaling the training data output
    scaler = StandardScaler()
    scaler.fit(train_y.numpy())
    train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()))

    ## Scaled data added to the training data
    train_data.add_output(output=train_y_scaled)

    return train_data, scaler

def train_data_multi_fidelity_noise(config, sampler, fun,):

    train_data = []
    for fid_no, (aug_coeff, samp_no) in enumerate(zip(
        config.aug_coeffs, [int(config.sampler.number_of_lf_samples), int(np.ceil(config.sampler.number_of_hf_samples))]
        )):
        
        ## Define augmentor to create artificial data fidelities
        noise_interpolator: f3dasm.functions.Augmentor = NoiseInterpolator(aug_coeff=aug_coeff, noise_var=1.)
        
        ## Generate training samples of the current fidelity with the sampler
        train_data_fid = sampler.get_samples(numsamples=samp_no)

        fun(train_data_fid) 

        ## Extracting output data for scaling
        train_y = torch.tensor(train_data_fid.get_output_data().values)

        ## Scaling the training data output (low fidelity only)
        if fid_no == 0:
            scaler = StandardScaler()
            scaler.fit(train_y.numpy())
        train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()))

        ## Augment the scaled training features
        train_y_scaled = noise_interpolator.augment(train_y_scaled.clone())

        ## Scaled data added to the training data
        train_data_fid.add_output(output=train_y_scaled)

        train_data.append(train_data_fid)

    return train_data, scaler