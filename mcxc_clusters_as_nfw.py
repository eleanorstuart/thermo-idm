import pandas as pd
import numpy as np
from astropy.table import QTable
from astropy import units as u
import pickle

from nfw_profile_heating import NFWProfile


def load_clusters(nrows=None, dataset='REFLEX'):
    skipfooter=0 if nrows else 1
    mcxccls=pd.read_csv('data/mcxc|.txt', header=3, sep='|', skiprows=[4],on_bad_lines='warn', skipfooter=skipfooter)
    mcxccls.columns=mcxccls.columns.str.strip()

    selected_cls=mcxccls[mcxccls['Sub-Cat'].str.strip()==dataset]

    cls_data={'M500':selected_cls['M500'],
          'L500':selected_cls['L500'],
          'R500':selected_cls['R500'],
          'z':selected_cls['R500']
         }
    units={
        'M500': 1e14*u.Msun,
        'L500': 1e37*u.W,
        'R500':u.Mpc,
        'z': None
    }

    cls_table=QTable(cls_data, units=units)

    with open ('data/'+dataset+'.p', 'rb') as fp:
        L_uncertainties=np.array(pickle.load(fp))
    L_uncertainties_conv=(L_uncertainties*1e37*u.W).to(u.erg / u.s).value

    n = nrows or len(L_uncertainties_conv)

    return [
        NFWProfile(
            cls_table['z'][i],
            M500=cls_table['M500'][i],
            R500=cls_table['R500'][i],
            L500=cls_table['L500'][i]
        )
        for i in range(n)
    ]