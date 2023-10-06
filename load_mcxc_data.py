import pandas as pd
from astropy.table import QTable
from astropy import units as u

from cluster import Cluster

def load_clusters(nrows=None):
    skipfooter=0 if nrows else 1
    mcxccls=pd.read_csv('data/mcxc|.txt', header=3, sep='|', skiprows=[4],on_bad_lines='warn', skipfooter=skipfooter, nrows=nrows)
    mcxccls.columns=mcxccls.columns.str.strip()
    cls_data={'M500':mcxccls['M500'],
          'L500':mcxccls['L500'],
          'R500':mcxccls['R500']
         }
    units={
        'M500': 1e14*u.Msun,
        'L500': 1e37*u.W,
        'R500':u.Mpc
    }

    cls_table=QTable(cls_data, units=units)

    return [
        Cluster(
            cls_table['R500'][i],
            cls_table['M500'][i],
            L500=cls_table['L500'][i],
            m500=cls_table['M500'][i],
        )
        for i in range(mcxccls.shape[0])
    ] # TODO: Implement variance