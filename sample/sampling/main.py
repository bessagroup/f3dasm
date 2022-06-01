# %%

from designofexperiments import DesignOfExperiments
from sobolsequence import SobolSequencing
from randomuniform import RandomUniform
from latinhypercube import LatinHypercube


# %%

doe = DesignOfExperiments()
sob = SobolSequencing(doe)
lhs = LatinHypercube(doe)
ran = RandomUniform(doe)
# %%
dim = 4
samp = 5
sob.sample(samp, dim)
lhs.sample(samp, dim)
ran.sample(samp, dim)

# %%
