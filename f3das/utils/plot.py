'''
Created on 2020-09-16 08:45:43
Last modified on 2020-09-17 17:02:05

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

# third-party
import numpy as np
from matplotlib import pyplot as plt


# object definition

class BarPlot(object):

    def __init__(self, width=.8):
        self.width = width
        # initialize empy vars
        self.n_x = None
        self.n_y = None
        self.width_each = None
        self.fig = None
        self.ax = None

    def plot(self, y, labels=(), tick_label=(), yerr=None, **kwargs):

        # initialization
        if not hasattr(y[0], '__iter__'):
            y = [y]
        self.n_x = len(y[0])
        self.n_y = len(y)
        self.width_each = self.width / self.n_y
        xx = np.arange(self.n_x)
        if not labels:
            labels = [None] * self.n_y
        if not yerr:
            yerr = [None] * self.n_y
        elif not hasattr(yerr[0], '__iter__'):
            yerr = [yerr]

        # plot
        self.fig, self.ax = plt.subplots()
        for w, y_, label, yerr_ in zip(self.gen_dist_weight(self.n_y), y, labels, yerr):
            self.ax.bar(xx + w * self.width_each, y_, width=self.width_each,
                        label=label, yerr=yerr_, **kwargs)

        # correct x labels
        self.ax.set_xticks(xx)
        if tick_label:
            self.ax.set_xticklabels(tick_label)

        return self

    def add_text(self, v, middle=True, text=(), str_format='.2f',
                 horizontalalignment='center', **kwargs):
        '''
        Parameters
        ----------
        v : array:
            Position of the text.
        middle : bool
            If True, then the value will be plotted in the middle. Otherwise,
            in the top.
        text : array (shape(text) = shape(v))
            If text to be plotted differs from position.

        '''

        # initialization
        if not hasattr(v[0], '__iter__'):
            v = [v]
        if len(text) and not hasattr(text[0], '__iter__'):
            text = [text]
        if not len(text):
            text = v
        xx = np.arange(self.n_x)
        str_format = '{:%s}' % str_format

        # add text
        for w, val, t in zip(self.gen_dist_weight(self.n_y), v, text):
            for xx_, val_, t_ in zip(xx, val, t):
                if not val_ or not t_:  # allows to jump bars
                    continue
                if middle:
                    val_ = np.array(val_) / 2
                self.ax.text(xx_ + w * self.width_each, val_, str_format.format(t_),
                             horizontalalignment=horizontalalignment, **kwargs)

    @staticmethod
    def gen_dist_weight(n):
        i = 0
        if n % 2:
            num = -(n - 1) / 2
        else:
            num = -1 / 2 - (n - 2) / 2
        while i < n:
            yield num
            num += 1
            i += 1
