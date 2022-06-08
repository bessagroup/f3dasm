# %%

import numpy as np
from f3dasm.src.designofexperiments import DesignOfExperiments
from f3dasm.sampling.sobolsequence import SobolSequencing
from f3dasm.sampling.randomuniform import RandomUniform
from f3dasm.sampling.latinhypercube import LatinHypercube

# %%

doe = DesignOfExperiments()
sob = SobolSequencing(doe)
lhs = LatinHypercube(doe)
ran = RandomUniform(doe)
# %%
dim = 4
samp = 5
np.random.seed(42)

sob.sample(samp, dim)
lhs.sample(samp, dim)
ran.sample(samp, dim)