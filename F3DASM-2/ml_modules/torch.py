from ..src.sim import ML
import numpy as np

class torch_it(ML):
    """

        SIMULATION-Module wrap for PyTorch

    """
    def __init__(self, in_data, model,**kwargs):

        """ Initialize """

        self.__name__ = 'PyTorch-Regression'                 # Name module
        super().__init__(in_data, model, **kwargs)                    # Initialize base-class

    def post_init(self):

        """ Implement post_init rule !!!"""

        self.out_var = ['epoch','train','test']    # Name output
        self.out = np.zeros((self.model.epochs,len(self.out_var)))   # Initialize numpy output

    def to_do(self,i):
        
        """ Implement to_do method !!!"""

        self.out [i,0] = i+1
        self.out [i,1] = self.model.train()
        self.out [i,2] = self.model.test()
        if self.verbose:
            print('epoch:',int(self.out[i,0]),'-> train_error:', '{:.5f}'.format(self.out[i,1]),'-> test_error:', '{:.5f}'.format(self.out[i,2]))
        
        


