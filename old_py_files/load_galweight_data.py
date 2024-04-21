import pandas as pd
from astropy import units as u
from astropy.table import QTable

from cluster import Cluster, temp_from_vdisp

def variance(err_neg, err_pos):
    n_temp = temp_from_vdisp(err_neg)
    p_temp = temp_from_vdisp(err_pos)
    return n_temp + p_temp

def load_clusters(nrows=None):
    # given an integer nrows, returns nrows Clusters generated from GalWeight cluster dataset
    galwcls=pd.read_csv('data/galwcls.dat', sep='|', header=None, nrows=nrows)
    cls_data = {'sig500': galwcls[:][8],
            'M500': galwcls[:][11],
            'r200': galwcls[:][13],
            'sig200':galwcls[:][15],
            'err_neg':galwcls[:][16],
            'err_pos':galwcls[:][17],
            'M200':galwcls[:][18]}
    units = {'sig500': u.km/u.s,
            'M500': u.Msun,
            'r200': u.Mpc,
            'sig200': u.km/u.s,
            'err_neg':u.km/u.s,
            'err_pos':u.km/u.s,
            'M200': u.Msun, }
    cls_table = QTable(cls_data, units=units)


    clusters = [Cluster(cls_table['r200'][i], cls_table['M200'][i], cls_table['sig200'][i], m500=cls_table['M500'][i]) for i in range(galwcls.shape[0])]
    variances = variance(cls_table['err_neg'], cls_table['err_pos'])
    return clusters, variances