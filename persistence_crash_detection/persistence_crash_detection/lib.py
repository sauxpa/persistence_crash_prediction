import numpy as np
import gudhi as gd
from scipy.integrate import trapz

#Warning this is not the most clever and efficient way to implement approximate landscapes.
def landscapes_approx(diag_dim, x_min, x_max, nb_steps, nb_landscapes):
    """
    Credit to http://bertrand.michel.perso.math.cnrs.fr/Enseignements/TDA/Tuto-Part4.html.
    """
    landscape = np.zeros((nb_landscapes,nb_steps))
    step = (x_max - x_min) / nb_steps
    #Warning: naive and not the best way to proceed!!!!!
    for i in range(nb_steps):
        x = x_min + i * step
        event_list = []
        for pair in diag_dim:
            b = pair[0]
            d = pair[1]
            if (b <= x) and (x<= d):
                if x >= (d+b)/2. :
                    event_list.append((d-x))
                else:
                    event_list.append((x-b))
        event_list.sort(reverse=True)
        event_list = np.asarray(event_list)
        for j in range(nb_landscapes):
            if(j<len(event_list)):
                landscape[j,i]=event_list[j]
    return landscape

def landscape_at_t(data, t, window=-1, dim=0, nbld=3, resolution=500, length_max=0.3):
    """
    data: numpy array of shape (d, T) where T: number of timesteps, d: number of dimensions.
    """
    if window < 0:
        # select all points up to t
        data_t = data[:, :t]
    else:
        # select points in a sliding window
        data_t = data[:, np.max([t-window, 0]):t]
        
    rips_complex = gd.RipsComplex(data_t, max_edge_length=length_max)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=dim+1)
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    
    persistence_intervals = simplex_tree.persistence_intervals_in_dimension(dim)
    landscapes = landscapes_approx(persistence_intervals, 0, length_max, resolution, nbld)

    return landscapes, np.arange(0, length_max, length_max / resolution)
