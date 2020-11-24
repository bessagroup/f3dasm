'''
Created on 2020-11-24 11:25:42
Last modified on 2020-11-24 12:25:21

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''


class MeshGenerator(object):

    def __init__(self):
        self.size = .02
        self.deviation_factor = .4
        self.min_size_factor = .4

    def change_definitions(self, **kwargs):
        '''
        See mesh definition at __init__ to find out the variables that can be
        changed.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def generate_mesh(self, part):
        # TODO: improve to control element size, etc
        part.seedPart(size=self.size,
                      deviationFactor=self.deviation_factor,
                      minSizeFactor=self.min_size_factor)

        part.generateMesh()
