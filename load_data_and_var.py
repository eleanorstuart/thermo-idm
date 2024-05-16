import pickle
import numpy as np
from astropy import units as u
from astropy.table import QTable

from nfw_profile_heating import NFWProfile

def load_data_and_var():
    # import datasets
    # this dataset contains the cluster properties (L500, M500, z, etc)
    with open('data/mcxc_psz_combined.pkl', 'rb') as f:
        comb_data = pickle.load(f)
    # this file contains the variances on the L500 measurements
    with open('data/REFLEX.p', 'rb') as g:
        errors_dict = pickle.load(g)

    # link the variances with the associated clusters
    reflex_data = comb_data[comb_data['Sub-Cat']=='REFLEX']
    reflex_appended = reflex_data.copy()
    reflex_appended['L_err'] = reflex_appended['OName'].apply(lambda x: errors_dict.get(x))
    reflex_clean = reflex_appended[reflex_appended['L_err'].notna()]

    L_err_list = reflex_clean['L_err'].tolist()
    variances = (np.array(L_err_list)*1e37*u.W).to(1e44*u.erg/u.s)

    return reflex_clean, variances


def generate_nfws_from_combined_dataset(dataset, nrows=None):
    cls_data={'M500':dataset['MSZ'],
          'L500':dataset['L500'],
          'z':dataset['z_y'] # take z from Planck
         }
    units={
        'M500': 1e14*u.Msun,
        'L500': 1e37*u.W,
        'z': None
    }

    cls_table=QTable(cls_data, units=units)
    
    n=nrows or len(cls_table)
    
    return [
        NFWProfile(
            cls_table['z'][i],
            M500=cls_table['M500'][i],
            L500=cls_table['L500'][i],
        )
        for i in range(n)
    ]