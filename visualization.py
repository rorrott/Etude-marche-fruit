import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import scipy.stats as st 
from scipy import stats
from scipy.stats import probplot

def display_scree_plot(pca, n_comp=None):
    
    if n_comp == None: # If no n_comp is provided, use all components
        num_components = len(pca.explained_variance_ratio_) # n_comp is provided
    elif n_comp < len(pca.explained_variance_ratio_):
        num_components = n_comp
    else: #If the n_comp provided is greater than the total number of components, then use all components
        num_components = len(pca.explained_variance_ratio_)
    indices = np.arange(num_components)
    values = pca.explained_variance_ratio_
    
    values = values[:num_components] 
    
    plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    # Create array of cumulative variance explained for each n^th component
    cumulative_values = np.cumsum(values)
    # Plot bar chart of variance explained vs each component
    ax.bar(indices, values, color='tab:red')
    # Plot line chart of cumulative variance explained vs number of components
    ax.plot(indices, cumulative_values, c="green",marker='o')
     
    # Plot the annotations only if there are less than 21 components, else it gets messy
    if num_components <= 20:
        for i in range(num_components):
            ax.annotate(r"%s%%" % ((str(values[i]*100)[:4])), (indices[i]+0.2, values[i]), va="bottom", ha="center", fontsize=12)
        
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
    
    #scree = pca.explained_variance_ratio_*100
    #plt.bar(np.arange(len(scree))+1, scree)
    #plt.plot(np.arange(len(values)), values.cumsum(),c="red",marker='o')
    #plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.savefig('pca.png')     
    plt.show(block=False)
    
    
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
            
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            
            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            #plt.savefig('circle1.png')
            #plt.savefig('circle2.png')
            plt.show(block=False)   
            
            
            
            
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            
            plt.gca().spines['bottom'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            #plt.savefig('factoriel1.png')
            plt.show(block=False)  
 
#################### analyse

def get_columns(data):
    # Extract columns
    s = data.dtypes == 'object'
    object_cols = data.columns[s].tolist()
    
    # Extract num columns
    n = (data.dtypes == 'int64') | (data.dtypes == 'float64')
    numerical_cols = data.columns[n].tolist()
    
    return object_cols, numerical_cols

#object_cols, numerical_cols = get_columns(region_mean)


#############
def check_normality(data, col, label_txt, conclusion = None, color='blue'):
    fig = plt.figure(facecolor = 'whitesmoke', figsize=(20, 5), dpi=100)
    
    ax_left = fig.add_axes([0, 0, .2, 1], facecolor='whitesmoke')
    ax_left.axis('off')
    ax_left.text(.4, .9, label_txt, color='red', weight='bold', size=25)
    ax_left.text(.1, .8, 'Skewness {:.2f}'.format(stats.skew(data[col], bias=False), size=30))
    ax_left.text(.1, .7, 'Kurtosis {:.2f}'.format(stats.kurtosis(data[col], bias=False), size=30))
    #ax_left.text(.1, .6, 'Shapiro {:.2f}'.format(stats.shapiro(data[col]), size=30))
    shapiro = stats.shapiro(data[col])
    ax_left.text(.1, .6, 'Shapiro {:.2f}'.format(shapiro[1]))
    #ax_left.text(.1, .6, 'Shapiro : \n{}'.format(stats.shapiro(data[col])))
    
    # Histogram
    ax1 = fig.add_axes([.25, 0, 0.3, .8], facecolor='whitesmoke')
    sns.distplot(data[col], color=color, kde=True, ax=ax1)
    ax1.set_xlabel(label_txt, fontsize=16)
    ax1.set_ylabel('frequency', fontsize=16)
    ax1.set_title('Histogram',  color='crimson', fontsize=18, weight='bold')
    ax1.tick_params(labelsize=10)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    ax1.grid(False)
    
    # QQ plot
    ax2 = fig.add_axes([.58, 0, .3, .8], facecolor='whitesmoke')
    probplot(data[col], plot=ax2)
    ax2.set_xlabel('Theoritical Quantiles', fontsize=16)
    ax2.tick_params(labelsize=10)
    ax2.set_title('QQ-Plot', color='crimson', fontsize=18, weight='bold')
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    ax2.grid(False)
    plt.show()


#########################
def plot_scatter(data, x, xlab, y, ylab):
    """inspect the relationship between specified variables"""
    fig,ax = plt.subplots(1,1,figsize=(15,5),dpi=100,facecolor="whitesmoke")
    sns.regplot(data=data,x=x,y=y,color='blue',ax=ax)
    ax.set_xlabel(xlab,fontsize=14)
    ax.set_ylabel(ylab,fontsize=14)
    ax.set_facecolor('whitesmoke')
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

