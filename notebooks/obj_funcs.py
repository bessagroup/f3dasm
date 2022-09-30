#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:31:24 2022

@author: surya

Aim:  To reparameterize benchmark functions using neural networks
Models : NN models - In tensorflow
Objective functions :  Autograd numpy [This file]

What I need to do:
    1. Model outputs the inputs to the objective function
    2. Calculate function value
    3. Calculate the gradients and update the model's weights
    4. This is repeated until convergence
    
Experiments:
    1. Test on functions in 2D
    2. Test on higher dimensionalities
    3. Test with the various optimizers (GD, HD, GD_LS, LBFGS...)
    4. Analyze the results using availble tools
    
"""
#%%
import autograd.numpy as np

import SALib.sample.latin as slb

DIMENSIONALITY = 2
class Function:
    """
    Main class to initialize a test function
    """
    def __init__(self, seed, name, orig_bounds=[0.0, 1.0], char=[],
                 dim=DIMENSIONALITY, global_minval =0.0, global_min =[]):
        
        self.name = name
        "Name of the function: f + seed."

        self.dim = dim
        "Dimensionality of the testproblem. Default is 2."

        self.pop = self.dim * 2 + 1  # arbitrary
        "Population size of the initial solutions."

        self.bounds = [0.0, 1.0]  # normalized bounds
        "Normalized box-constrained bounds of the testproblem."
        
        self.global_minval = global_minval
        "Value of the global minimum of the function"
        
        self.global_min = global_min#[np.zeros((1,self.dim))]
        "The location(s) of the global minimum; At least one"

        self.orig_bounds = orig_bounds

        self.seed = seed
        "Random number generator seed."
        np.random.seed(seed=seed)

        self.reroll_counter = 0
        "Takes track of how many times rerolls are requested"

        self.fevals = 0
        "Takes track of the number of function evaluations"
        
        self.o = np.random.uniform(0.3, 0.5, self.dim)  # offset
        "Input parameter off-set."

        self.char = char
        "List of features on the testproblem."

        #self.getx0()        

    def reset_seed(self):
        np.random.seed(seed=self.seed)
        # self.reroll_counter = 0
        self.fevals = 0
        self.setx0(forcereset=True)

    def get_seed(self):
        return self.seed
    
    def offset(self, x):
        return x - self.o

    def denormalize(self, x):
        return (self.orig_bounds[1] - self.orig_bounds[0]) * x + self.orig_bounds[0]
    
    def normalize(self,x):
        return (x - self.orig_bounds[0])/ (self.orig_bounds[1] - self.orig_bounds[0])

    def ask(self, x, skip_fevals=False):
        x = self.offset(x)
        x = self.denormalize(x)
        y = self.func(x)
        if not skip_fevals:
            self.fevals += 1
        return y

    def makedictx0(self):
        "Make dictionary for SALib.sample"
        names = []
        b = []
        for i in range(self.dim):
            names.append("x%i" % i)

        for j in range(self.dim):
            b.append(self.bounds)

        problem = {"num_vars": self.dim, "names": names, "bounds": b}
        return problem

    def getx0(self):
        "Latin hypercube sampling using SALib.sample"
        problem = self.makedictx0()
        self.x0 = slb.sample(problem, self.pop, seed=self.seed)

        "Compute objective values of x0"
        self.y0 = np.array([self.ask(i) for i in self.x0])
        np.random.seed(self.seed)
        self.fevals = 0  
        
    def find_global_min(self, px = 300, scale_y = False):
        """
        Creates a grid and calculates teh approximate global minimum of a given function
        Inputs:
            px : Integer --> Grid size along one direction 
                [Default =300 implies a 300 x 300 grid creation]
            scale_y : Boolean --> Whether to normalize the Y values or not
        Outputs:
            global_min_loc: Tuple --> coordinates of the global minimum (approximate)
            X1, X2 : Numpy 2D arrays --> Mesh coordinates of variables x1 and x2
            Y : Numpy 2D array -->  Normalized function values at (x1,x2)
        """
        
        X1 = np.linspace(0, 1, num=px)
        X2 = np.linspace(0, 1, num=px)
        X1, X2 = np.meshgrid(X1, X2)
    
        Y = np.zeros([len(X1), len(X1)])
    
        for i in range(len(X1)):
            for j in range(len(X1)):
                xy = np.array([X1[i, j], X2[i, j]])
                Y[i, j] = self.ask(xy)
    
        self.global_minval_approx = Y.min()    
        # normalize Y
        if scale_y:
            Ymin = Y.min()
            Ymax = Y.max()
            for i in range(len(X1)):
                for j in range(len(X1)):
                    xy = np.array([X1[i, j], X2[i, j]])
                    Y[i, j] = (Y[i, j] - Ymin) / (Ymax - Ymin)
        
        Ym = np.unravel_index(Y.argmin(), Y.shape)
        
        global_min_loc = (X2[Ym[1], 0], X1[0,Ym[0]]) #TODO: To check this -- done
        self.global_min_loc_approx = global_min_loc
         
        return global_min_loc, X1, X2, Y
        
 

    def plot3d(self, data = None, px=100, is_log = False, scale_y = False):
    
        """
        Generate a 3D plot of a slice of the 4D function
        px = number of evaluations in one dimension for plotting the response surface
        """
        # import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcol
        import matplotlib
        fevals_before = self.fevals
        if is_log:
            scale = lambda x: np.log(x)
        else:
            scale = lambda x: x
            
        g_min, X1, X2, Y = self.find_global_min(px = px, scale_y = scale_y)   
    
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        ax = plt.axes(projection="3d", elev=50, azim=-50, computed_zorder=False)  # elev=50, azim=-50
    
        norm = mcol.LogNorm()
   
        ax.plot_surface(
            X1,
            X2,
            scale(Y),
            rstride=1,
            cstride=1,
            edgecolor="none",
            alpha=0.8,
            cmap="viridis",
            norm=norm,            
            zorder=1,
        )  # 0.8
        if data is not None:
            NPOINTS= data.shape[0]
            assert data.shape[1] == 2            
            cmap = matplotlib.cm.get_cmap('hot')
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=NPOINTS)
            colors = [cmap(normalize(value)) for value in range(NPOINTS)]
            path_x = data[:,0]
            path_y = data[:,1]
            fn_val = np.array([self.ask(np.array(x)) for x in zip(path_x,path_y)])
            #fn_val_norm = self.normalize(fn_val)
            
            #As0 = plt.plot(path_x,path_y, color="k",markersize=0, alpha=1,zorder=8)
            As = ax.scatter3D(path_x[1:-1],path_y[1:-1],fn_val[1:-1], color=colors[1:-1][::-1],
                              edgecolor='k',alpha=1, s=50, zorder=2)
            As1 = ax.scatter3D(path_x[0], path_y[0], fn_val[0], facecolor='k', edgecolor='k',
                                marker='v', alpha=1,s=50, zorder=2, label = "Start")
            As1 = ax.scatter3D(path_x[-1],path_y[-1], facecolor='r', 
                                marker='X', alpha=1,s=50,zorder=2, label ="End")
            
        ax.set_xlabel('$X_{1}$',fontsize='small') #'x-large'
        ax.set_ylabel('$X_{2}$',fontsize='small')
        ax.set_zlabel('$f(X)$',fontsize='small')
        plt.title(self.name + " function")
        ax.set_xlim((self.bounds[0], self.bounds[1]))
        ax.set_ylim((0.0, 1.0))
        ax.scatter3D(
            g_min[0],
            g_min[1], self.global_minval_approx,
            color="white",
            edgecolors="black",
            marker="X",
            s=1.5 * px,
            label="Global optimum",
            alpha=0.8,
        )  # 0.55*px
        ax.set_xticks(np.linspace(self.bounds[0], self.bounds[1], 11))
        ax.set_yticks(np.linspace(self.bounds[0], self.bounds[1], 11))
        ax.legend()
        #ax.set_zticks(np.linspace(0.0, 1.0, 11))   

        self.fevals = fevals_before
        return fig, ax
    
    def plot_contour(self, data=None, px = 50, is_log = False, scale_y = False, zoom = False):
        """
        For plotting the (filled) contour plot for the given function having 2 variable inputs        
        Inputs:
            data : Numpy array -->data is provided in the form  [[x1,y1], [x2,y2].....]
            px : Integer --> Number of grid points to use
            is_log: Boolean --> Whether to plot the function values in log scale or not
            scale_y: Boolean --> Whether to normalize the function values or not
            zoom: Boolean --> Whether to zoom near the data points (by 20%)
        
        """
        
        import matplotlib.pyplot as plt
        import matplotlib 
        fevals_before = self.fevals
        if is_log:
            scale = lambda x: np.log(x)
        else:
            scale = lambda x: x
            
        g_min, X1, X2, Y = self.find_global_min(px = px, scale_y = scale_y)
        fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k',
                                 frameon=False)
        CS = plt.contourf(X1, X2, scale(Y), 100, zorder=0)        
        plt.colorbar()
        if data is not None:
            NPOINTS= data.shape[0]
            assert data.shape[1] == 2            
            cmap = matplotlib.cm.get_cmap('hot')
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=NPOINTS)
            colors = [cmap(normalize(value)) for value in range(NPOINTS)]
            path_x = data[:,0]
            path_y = data[:,1]
            As0 = plt.plot(path_x,path_y, color="k",markersize=0, alpha=1,zorder=8)
            As = plt.scatter(path_x,path_y,color=colors[::-1],edgecolor='k',alpha=1,zorder=10)
            As1 = plt.plot(path_x[0],path_y[0], markerfacecolor='k', markeredgecolor='k',
                                marker='v', markersize=10, alpha=1,zorder=10, label = "Start")
            As1 = plt.plot(path_x[-1],path_y[-1], markerfacecolor='r', markeredgecolor='r',
                                marker='X', markersize=10, alpha=1,zorder=10, label ="End")
            if zoom:
                minx, maxx = path_x.min(), path_x.max()
                miny, maxy = path_y.min(), path_y.max()
                zoom_lim = 0.02 # 20% margin for zooming
                plt.xlim(minx*0.8 , maxx*1.2 )
                plt.ylim(miny*0.8, maxy*1.2)

        plt.scatter(
            g_min[0],
            g_min[1],
            color="white",
            edgecolors="black",
            marker="X",
            s=1.5 * px,
            label="Global optimum",
            alpha=0.8,
        )  # 0.55*px
        plt.xlabel("$X_{1}$", fontsize=16)  # 20
        plt.ylabel("$X_{2}$", fontsize=16)
        #ax.yaxis.set_label_position("right")
        plt.legend(fontsize="small", loc="lower right")
        plt.show()#S:to see
        self.fevals = fevals_before
        

# Create a list with all functions to choose from"
funclist = []
# ......................................
"""
Benchmark optimization functions
from https://www.sfu.ca/~ssurjano/optimization.html
"""
# ......................................


class Levy(Function):
    def __init__(self, seed, dim):

        orig_bounds = [-10.0, 10.0]
        char = ["multimodal"]
        name = "levy"
        global_min = np.ones((1,dim))
        global_minval = 0
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char,
                         dim =dim,global_minval=global_minval, global_min = global_min)
        self.global_min = self.normalize(self.global_min)
        
    def func(self, x):
        z = 1 + (x - 1) / 4
        c = (
            np.sin(np.pi * z[0]) ** 2
            + sum((z[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1) ** 2))
            + (z[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * z[-1]) ** 2)
        )
        return c


funclist.append(Levy)

# ......................................


class Ackley(Function):
    def __init__(self, seed, dim):
        orig_bounds = [-40.0, 40.0]
        char = ["multimodal", "steep"]
        name = "ackley"
        global_min = np.zeros((1,dim))
        global_minval = 0
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char,
                         dim =dim,global_minval=global_minval, global_min = global_min)
        self.global_min = self.normalize(self.global_min)

    def func(self, x, a=20, b=0.2, c=2 * np.pi):
        n = len(x)

        s1 = sum(x**2)
        s2 = sum(np.cos(c * x))
        cc = -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)
        return cc


funclist.append(Ackley)

# ......................................


class Rosenbrock(Function):
    def __init__(self, seed, dim):
        orig_bounds = [-5.0, 10.0]
        char = ["unimodal"]
        name = "rosenbrock"
        global_min = np.ones((1,dim))
        global_minval = 0
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char,
                         dim =dim,global_minval=global_minval, global_min = global_min)
        self.global_min = self.normalize(self.global_min)

    def func(self, x):

        x0 = x[:-1]
        x1 = x[1:]
        c = sum((1 - x0) ** 2) + 100 * sum((x1 - x0**2) ** 2)
        return c


funclist.append(Rosenbrock)

# ......................................


class Schwefel(Function):
    def __init__(self, seed, dim):
        orig_bounds = [-500.0, 500.0]
        char = ["multimodal"]
        name = "schwefel"
        global_min = np.ones((1,dim))*420.96
        global_minval = 0
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char,
                         dim =dim,global_minval=global_minval, global_min = global_min)
        self.global_min = self.normalize(self.global_min)

    def func(self, x):
        n = len(x)

        c = 418.9829 * n - sum(x * np.sin(np.sqrt(abs(x))))
        return c


funclist.append(Schwefel)

# ......................................


class Rastrigin(Function):
    def __init__(self, seed, dim):
        orig_bounds = [-5.12, 5.12]
        char = ["multimodal"]
        name = "rastrigin"
        global_min = np.zeros((1,dim))
        global_minval = 0
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char,
                         dim =dim,global_minval=global_minval, global_min = global_min)
        self.global_min = self.normalize(self.global_min)

    def func(self, x):
        n = len(x)
        c = 10 * n + sum(x**2 - 10 * np.cos(2 * np.pi * x))
        return c


funclist.append(Rastrigin)



class Styblinski(Function):
    def __init__(self, seed, dim):
        orig_bounds = [-5.0, 5.0]
        char = ["multimodal"]
        name = "styblinski"
        global_min = np.ones((1,dim))* -2.903534
        global_minval = -39.16599*dim
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char,
                         dim =dim,global_minval=global_minval, global_min = global_min)
        self.global_min = self.normalize(self.global_min)

    def func(self, x):

        c = 0.5 * sum(x**4 - 16 * x**2 + 5 * x)
        return c


funclist.append(Styblinski)

# ......................................


class Branin(Function):  # 2D
    def __init__(self, seed, name):
        orig_bounds = [0.0, 15.0]
        char = ["multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):

        x1 = x[:-1] + 5.0  # correct for the uneven box
        x2 = x[1:]

        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        cc = sum(
            a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        )
        return cc


funclist.append(Branin)


# ......................................


class SchafferF6(Function):  # 2D
    def __init__(self, seed, name):
        orig_bounds = [-100.0, 100.0]
        char = ["steep", "multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):

        x1 = x[:-1]
        x2 = x[1:]

        x1x2 = x1**2 + x2**2

        c = sum(0.5 + (np.sin(np.sqrt(x1x2)) ** 2 - 0.5) / ((1 + 0.001 * x1x2) ** 2))
        return c


funclist.append(SchafferF6)

# ......................................


class Beale(Function):  # 2D
    def __init__(self, seed, name):
        orig_bounds = [-4.5, 4.5]
        char = ["multimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):

        x1 = x[:-1]
        x2 = x[1:]

        c = sum(
            (1.5 - x1 + x1 * x2) ** 2
            + (2.25 - x1 + x1 * x2**2) ** 2
            + (2.625 - x1 + x1 * x2**3) ** 2
        )
        return c


funclist.append(Beale)

# ......................................


class AckleyNo2(Function):
    def __init__(self, seed, name):
        orig_bounds = [-4.0, 4.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)

        x1 = x[:-1]
        x2 = x[1:]
        cc = sum(-200 * np.exp(-0.2 * np.sqrt(x1**2 + x2**2)))
        return cc


funclist.append(AckleyNo2)

# ......................................


class Bohachevsky(Function):
    def __init__(self, seed, name):
        orig_bounds = [-100.0, 100.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)

        x1 = x[:-1]
        x2 = x[1:]
        cc = sum(
            x1**2
            + 2 * x2**2
            - 0.3 * np.cos(3 * np.pi * x1)
            - 0.4 * np.cos(4 * np.pi * x2)
            + 0.7
        )
        return cc


funclist.append(Bohachevsky)

# ......................................


class Matyas(Function):
    def __init__(self, seed, name):
        orig_bounds = [-10.0, 10.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)

        x1 = x[:-1]
        x2 = x[1:]
        cc = sum(0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2)
        return cc


funclist.append(Matyas)

# ......................................


class Zakharov(Function):
    def __init__(self, seed, name):
        orig_bounds = [-5.0, 10.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)
        cc = (
            sum(x**2)
            + sum(0.5 * np.arange(1, len(x) + 1) * x) ** 2
            + sum(0.5 * np.arange(1, len(x) + 1) * x) ** 4
        )
        return cc


funclist.append(Zakharov)

# ......................................


class McCormick(Function):
    def __init__(self, seed, name):
        orig_bounds = [-3.0, 4]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)
        x1 = x[:-1]
        x2 = x[1:]

        cc = sum(np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1)
        return cc


funclist.append(McCormick)

# ......................................


class Leon(Function):
    def __init__(self, seed, name):
        orig_bounds = [-5.0, 5.0]
        char = ["unimodal"]
        super().__init__(seed, name, orig_bounds=orig_bounds, char=char)

    def func(self, x):
        n = len(x)
        x1 = x[:-1]
        x2 = x[1:]

        cc = sum(100 * (x2 - x1**3) ** 2 + (1 - x1) ** 2)
        return cc


funclist.append(Leon)

# ......................................

if __name__ == "__main__":
    "This chooses a function based on a seed and plots the response surface"
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)
#%%