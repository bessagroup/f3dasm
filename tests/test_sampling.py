# -*- coding: utf-8 -*-

import numpy as np
import pytest
from f3dasm.sampling import randomuniform, latinhypercube, sobolsequence
from f3dasm.src.designofexperiments import DesignOfExperiments

def test_randomuniform_samples():
    ran = randomuniform.RandomUniform(DesignOfExperiments(), seed=42)
    samples = ran.sample(numsamples=3, dimensions=2)
    truth = np.array([[0.37454012, 0.95071431], [0.73199394, 0.59865848], [0.15601864, 0.15599452]])
    assert samples == pytest.approx(truth)

def test_latinhypercube_samples():
    lhs = latinhypercube.LatinHypercube(DesignOfExperiments(), seed=42)
    samples = lhs.sample(numsamples=3, dimensions=2)
    print(samples)
    truth = np.array([[0.57733131, 0.71866484], [0.12484671, 0.53288616], [0.71867288, 0.31690477]])
    assert samples == pytest.approx(truth)

def test_sobolsequence_samples():
    sob = sobolsequence.SobolSequencing(DesignOfExperiments(), seed=42)
    samples = sob.sample(numsamples=3, dimensions=2)
    print(samples)
    truth = np.array([[0., 0.], [0.5, 0.5], [0.75, 0.25]])
    assert samples == pytest.approx(truth)

if __name__ == '__main__':
    pytest.main()
