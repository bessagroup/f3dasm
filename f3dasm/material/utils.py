'''
Created on 2020-04-08 11:40:11
Last modified on 2020-04-08 11:41:04
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

@collaborators:
    F. Otero (fotero@inegi.up.pt)

Main goal
---------
Define objects common to the modules of this subpackage.
'''


class Property(object):

    def __init__(self, value, deviation=None, unit=None, info=None):
        if value is None:
            self.value = None
        elif type(value) is list:
            self.value = value
        else:
            self.value = float(value)
        self.deviation = None if deviation is None else float(deviation)
        self.unit = unit
        self.info = info

    @property
    def data(self):
        data = {'value': self.value,
                'deviation': self.deviation,
                'unit': self.unit,
                'info': self.info}
        return data
