import numpy as np

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.colorbar as mpl_colorbar

from typing import Literal


def hist2D_V2(x, y, bins=20, cmap='inferno', zero_color='black',new_fig=True,vmin=1,**np_hist_kwargs):
    if new_fig : plt.figure()

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins,**np_hist_kwargs)
    
    custom_cmap = plt.get_cmap(cmap)  
    custom_cmap.set_under(zero_color) 
    
    # --- plotting ---
    plt.imshow(
        hist.T,
        extent=(xedges.min(), xedges.max(), yedges.min(), yedges.max()),
        cmap=custom_cmap,
        origin='lower',
        aspect='auto',
        vmin=vmin)
    
    plt.colorbar(label='Occurences')
 


def remove_table_title():
    l_table = plt.gca().tables
    
    if len(l_table) == 0 : raise ValueError("No table found")
    
    T = l_table[0]
    import matplotlib.table as MT
    T:MT.Table
    nb_lignes = max( cell[0] for cell in T.get_celld())
    T.remove()
    title_str = plt.gca().title.get_text()
    for _ in range(nb_lignes+1):
        if title_str[-1] == '\n':
            title_str = title_str[:-1]
            
    plt.title(title_str)

def table_title(data:list[list]):
    """
    Args
    ----
    - `data` : list[list]
    """
    N = len(data)
    add_to_title("\n"*N,new_line=False)
    plt.table(cellText=data,loc='top')

def hist2D(
        X:np.ndarray,
        Y:np.ndarray,
        bins:int=20,
        new_fig:bool=True):
    if new_fig : plt.figure()
    
    plt.hist2d(x=X, y=Y, bins=bins, edgecolors='#000')
    plt.colorbar(label='Number of values')

def add_to_title(txt:str,new_line:bool=True):
    """
	Args
	----
	- `txt` : str
	- `new_line` : bool (default = True)

	"""
    if new_line : txt = '\n'+txt
    Ax = plt.gca() # get current ax
    Ax:plt.Axes
    old_title = Ax.get_title()
    plt.title(old_title + txt)

def plot_vertical_line(x:float,color='#f00',linestyle='--',**plot_kwargs):
	"""
	Args
	----
	- `x` : float
	- `color` : str | Any
	- `linestyle` : str
	- `**plot_kwargs` -> redirected to plt.plot()
	"""
	Y = plt.ylim()
	plt.plot([x]*2,Y,scaley=False, color=color,linestyle=linestyle,**plot_kwargs)

def plot_horizontal_line(y:float,color='#f00',linestyle='--',**plot_kwargs):
	"""
	Args
	----
	- `y` : float
	- `color` : str | Any
	- `linestyle` : str
	- `**plot_kwargs` -> redirected to plt.plot()
	"""
	X = plt.xlim()
	plt.plot(X,[y]*2,scalex=False,color=color,linestyle=linestyle,**plot_kwargs)

def add_diagonal_line(color='#f00',linestyle='--',**plot_kwargs):
    """
    Args
    ----
    - `color` : matplotlib color (default = '#f00')
    - `linestyle` : matplotlib linestyle (default = '--')
    - `**plot_kwargs` 
    """
    xmin,xmax = plt.xlim()
    ymin,ymax = plt.ylim()
    line_min = max(xmin,ymin)
    line_max = min(xmax,ymax)
    plt.plot([line_min,line_max],[line_min,line_max],
             scalex=False,
             scaley=False,
             color=color,
             linestyle=linestyle,
             **plot_kwargs)

def mutiple_plot_v2(l_Y:list|np.ndarray,
					l_X:list|np.ndarray|None=None,
					color_range:tuple=("#f00","#00f"),
					colorbar_label:str='',
					colorbar_lim:tuple[float]=(0,1),
					new_fig:bool=True):
	'''
	Args
	----
	`l_Y` : list | Array
	`l_X` : list | Array | None
	`color_range`		: tuple 		(default = ("#f00","#00f"))
	`colorbar_label` 	: str 			(default = '')
	`colorbar_lim` 		: tuple[float] 	(default = (0,1))
	`new_fig` 			: bool 			(default = True)
	'''
	N = len(l_Y)
	if new_fig : plt.figure()
	l_color = colorFader(*color_range,N=N)

	# --- creation of the color bar ---
	cmap = colors.ListedColormap(l_color)
	mappable = cm.ScalarMappable(cmap=cmap)
	cbar = plt.colorbar(mappable,ax=plt.gca(),values=np.linspace(*colorbar_lim,N))
	cbar : mpl_colorbar.Colorbar # only to help the autocompletion
	cbar.ax.set_ylabel(colorbar_label)

	if l_X is None : 
		for i,Y in enumerate(l_Y):
			color = l_color[i]
			plt.plot(Y,color=color)
	else :
		for i,(Y,X) in enumerate(zip(l_Y,l_X)):
			color = l_color[i]
			plt.plot(X,Y,color=color)

def mutiple_plot(l_Y:list|np.ndarray,
				 color_range:tuple=("#f00","#00f"),
				 N_step:int=5,
				 plot_step:int=1,
				 new_fig:bool=True):
	'''
	Args
	----
	`l_Y` : list | Array
	`color_range` : tuple (default = ("#f00","#00f"))
	`N_step` : int (default = 5)
	`plot_step` : int (default = 1)
	`new_fig` : bool (default = True)

	'''
	step = (N_step+len(l_Y))//N_step
	if new_fig : plt.figure()
	l_color = colorFader(*color_range,N=N_step)
	for i,Y in enumerate(l_Y):
		color = l_color[i//step]
		if i%plot_step == 0 : # else skip the plot
			plt.plot(Y,color=color)
		
def colorFader(c1:str,c2:str,N:int=5): 
	'''
	Args
	----
	`c1` : str
	`c2` : str
	`N` : int (default = 5)
		Number of color to give

	Returns
	-------
	`l_color` : list
	'''
	c1=np.array(mpl.colors.to_rgb(c1))
	c2=np.array(mpl.colors.to_rgb(c2))
	mix=np.linspace(0,1,N)
	l_color = []
	for i in np.linspace(0,1,N,endpoint=True):
		l_color.append(mpl.colors.to_hex((1-i)*c1 + i*c2))
	return l_color

def use_cmap(cmap:Literal['gray','bwr','hsv','brg']='gray'):
	"""
	Args
	----
	cmap : str (default = 'gray')
		possibilities :
			- gray
			- bwr (Blue White Red)
			- hsv (Red Orange Yellow Green Blue Purple Red)
			- brg (Blue Red Green)
	"""
	plt.rcParams['image.cmap'] = cmap

def use_specific_plot_size(fig_size:tuple[float]=(9, 6)):
	"""
	Args
	----
	- `fig_size` : tuple[float] (default = (9,6))
	"""
	mpl.rcParams['figure.figsize'] = fig_size

def use_big_label():
	mpl.rcParams['axes.labelsize']='x-large'
	mpl.rcParams['legend.fontsize']='x-large'
	mpl.rcParams['xtick.labelsize']='x-large'
	mpl.rcParams['ytick.labelsize']='x-large'
	mpl.rcParams['ytick.labelsize']='x-large'
	mpl.rcParams['axes.titlesize']='xx-large'
		
def restore_default_param():
	mpl.rcdefaults()
	
def no_y_ticks():
	plt.yticks(ticks=[],labels=[])
	
def no_x_ticks():
	plt.xticks(ticks=[],labels=[])

def change_grid(step_x:int=None,step_y:int=None):
	"""Change the grid steps used by matplotlib
	
	Args
	----
	- `step_x` : int (default = None)
		- step used between 2 consecutives lines in the x direction
		- if None, it doesn't affect the grid

	- `step_y` : int (default = None)
		- step used between 2 consecutives lines in the y direction
		- if None, it doesn't affect the grid
	"""
	ax = plt.gca()
	ax : plt.Axes
	if step_x is not None : ax.xaxis.set_major_locator(ticker.MultipleLocator(step_x))
	if step_y is not None : ax.yaxis.set_major_locator(ticker.MultipleLocator(step_y))
	plt.grid(True)

def no_axis():
	plt.axis(False)
 
 
 
 