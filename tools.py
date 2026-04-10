import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 

# # Level 2 (Granule)
def time_from_attr2(ds):
    """Attach scalar time coordinate from time_coverage_start"""
    
    datetime = ds.attrs["time_coverage_start"].replace("Z", "")
    time = np.datetime64(datetime)
    return ds.assign_coords(time=time)

def open_L2_CLOUD_GPC2(path):
    
    prod = xr.open_dataset(path)
    gphy = xr.open_dataset(path, group="geophysical_data").squeeze()
    geo  = xr.open_dataset(path, group="geolocation_data").set_coords(["latitude", "longitude"])

    ds = xr.merge((prod, gphy, geo))

    ds = ds[[
            "cloud_bow_droplet_effective_radius",
            "cloud_bow_droplet_effective_variance",
            "cloud_bow_droplet_number_concentration_adiabatic"]]

    return ds


# Level 3
def time_from_attr(ds):
    """
       ds: a dataset corresponding to one or multiple Level-2 granules
    """
    datetime = ds.attrs["time_coverage_start"].replace("Z", "")
    time = np.datetime64(datetime, "ns")
    lat = ds['latitude']
    lon = ds['longitude']
    ds = ds.drop_vars(['latitude', 'longitude'])          # temporarily drop lat/lon
    ds = ds.expand_dims(time=[time])                      # expand time dimension for all remaining variables
    ds = ds.assign_coords(latitude=lat, longitude=lon)    # add lat/lon back
    return ds


def open_L2_CLOUD_GPC(path, bounding_box=(-170,-60,-120,-50)):
    """
    Args:
        path: a path to a dataset corresponding to one Level-2 granule
        bounding_box: optional argument of bounding box for region (defaults to Southern Ocean)
    Returns:
        the dataset restricted to the bounding_box region, with specific variables, and with the datetime added as time coord
    """

    prod = xr.open_dataset(path)
    gphy = xr.open_dataset(path, group="geophysical_data").squeeze()
    geo = xr.open_dataset(path, group="geolocation_data").set_coords(["latitude", "longitude"])
    ds = xr.merge((prod, gphy, geo))

    ds = ds[['cloud_bow_droplet_effective_radius',
             'cloud_bow_droplet_effective_variance',
             'cloud_bow_droplet_number_concentration_adiabatic',
             'cloud_bow_liquid_water_path']]
    W,S,E,N = bounding_box
    lat_mask = np.logical_and(ds.latitude<=N, ds.latitude>=S)
    lon_mask = np.logical_and(ds.longitude<=E, ds.longitude>=W)
    mask = np.logical_and(lat_mask, lon_mask)
    ds = ds.where(mask, drop=True)
    return(ds)

def plot_hist(cdnc,lwp):
    '''
    Plots a histogram as in Goren et al 2025 given cloud droplet number concentration and liquid water path. 
    '''
    fig,ax = plt.subplots()
    
    xbins = np.logspace(0, 2.5, 50) # <- make a range from 10**xmin to 10**xmax
    ybins = np.logspace(1, 2.5, 50) # <- make a range from 10**ymin to 10**ymax
    
    h,xedge,yedge,im = ax.hist2d(cdnc, lwp, bins=(xbins, ybins))
    
    h_norm = h/h.sum(axis=1,keepdims=True)
    h_norm=np.transpose(h_norm)
    
    im = ax.pcolormesh(xedge, yedge, h_norm, cmap='turbo', vmax=0.055)
    fig.colorbar(im, shrink=0.75)
    ax.set_xlabel('CDNC', fontsize=15)
    ax.set_ylabel(r'LWP', fontsize=15)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
