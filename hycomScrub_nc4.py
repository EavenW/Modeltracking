
# coding: utf-8

# Getting HYCOM Plots

## First steps, accessing DaP and exploring data

# The goal for this is to develop a framework for getting model data 
# from the various sources. Then linking it to files that give the 
# time and location data for various tropical storm and nor'easter
# tracks. Thus, we can plot the temperatures/salinities on the track
# in time, along with what happens ahead of and behind the storms.
# Where applicable, this can be compared with glider data. 

import netCDF4
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
import sys

# ## Import functions
def offset_hour_2000(y,m,d,h):
    """Returns year,month,day string for an offset day"""
    from datetime import datetime, timedelta
    delta = datetime(y,m,d,h,0,0) - datetime(2000,1,1,0,0,0)
    return delta.total_seconds() / 3600

def ymdh(offset_hour):
    """Returns year-month-day time string for an offset day"""
    from datetime import datetime, timedelta
    start = datetime(2000,1,1,0,0,0)
    curr_date = start + timedelta(hours=offset_hour)
    return curr_date.strftime('%Y%m%d%H')

def vincenty(loc1,loc2):
    """Returns the Vincenty distance between two (lat,lon) locations in km"""
    from vincenty import vincenty
    loc1 = (loc1[0], loc1[1])
    loc2 = (loc2[0], loc2[1])
    return vincenty(loc1, loc2)

def nearest_neighbors(lat_index, lon_index):
    """Returns 8 neighborhood indexes(?) and self index in a list for given index"""
    neighborhood = []
    for i in range(-1,2):
        for j in range(-1,2):
            neighborhood.append([lat_index+i,lon_index+j])
    return neighborhood

def location_to_index(target_lat,target_lon,lats,lons):
    """Return index for nearest model output node"""
    #lons = dataset.lon[:]
    #lats = dataset.lat[:]
    import numpy as np
    if target_lon < min(lons):
        target_lon += 360
    xsize = lons.shape[0]
    ysize = lats.shape[0]
    min_d = None
    y, x = None, None
    found_lat, found_lon = None, None
    for j,lat in enumerate(lats):
        for i, lon in enumerate(lons):
            d = np.sqrt((lon-target_lon)**2 + (lat-target_lat)**2)
            if min_d == None or d < min_d:
                min_d = d
                x = i
                found_lat = lat
                y = j
                found_lon = lon
    return (y,x,found_lat,found_lon)

def interpSlice_nc4(loc, date_index, variable, data):
    """ Returns weighted average of temperature at a given location
        interp(target ,neighborhood, depth_index, variable, date_index)
        date_index from date in hours since 2000/1/1
        depth_index from the depth profile of HYCOM assymilated data
        variable is a string that calls the ID of a given variable
    """
    import netCDF4
    import numpy as np
    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]
    depth= data.variables['depth'][:]
    y, x, found_lat, found_lon = location_to_index(loc[0],loc[1],lats,lons)
    target = [y, x]
    neighborlist = nearest_neighbors(y,x)
    neighborhood = [ [lats[i[0]], lons[i[1]]] for i in neighborlist]
    dist_list = [vincenty(loc,neighbor) for neighbor in neighborhood]
    weight_list = np.array([1./(dist*dist) if dist > 0 else 1.0 for dist in dist_list])
    var_matrix = []
    for neighbor in neighborlist:
    	tmp = data.variables[variable][date_index, :, neighbor[0], neighbor[1]]
        var_matrix.append(np.squeeze(tmp))
    var_matrix = np.asarray(var_matrix)
    var_list = [np.average(var_matrix[:,i], weights=weight_list) for i,j in enumerate(depth)]
    return var_list

def tempcolumn_nc4(loc, date_index, variable, data):
    """ Returns weighted average of temperature at a given location
        interp(target ,neighborhood, depth_index, variable, date_index)
        date_index from date in hours since 2000/1/1
        depth_index from the depth profile of HYCOM assymilated data
        variable is a string that calls the ID of a given variable
    """
    import netCDF4
    import numpy as np
    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]
    depth= data.variables['depth'][:]
    y, x, found_lat, found_lon = location_to_index(loc[0],loc[1],lats,lons)
    #print loc[0],loc[1], found_lat, found_lon
    var = data.variables[variable]
    var_list = var[date_index, :, y, x]
    return var_list

def hycomScrubber(hycom_url,variable,location,date_index):
    	## Getting the HYCOM
    	# This is from the Navy's reanalysis data GLB files, it is
    	# seperated into odd dates which will need to be searched
    	# through to make this fully automatic. This will require
    	# knowing the whole database names and accessing the 'time'
    	# variable that is in hours since January 1, 2000 (oddly).
    	#hycom_url = 'http://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_91.2'
    import numpy as np
    from netCDF4 import Dataset
    data = Dataset(hycom_url)
    depth = data.variables['depth'][:]
    temps = tempcolumn_nc4(location, date_index, variable, data)
    return temps[:25], depth[:25]

def hurrtimeconv(date):
    from datetime import datetime, timedelta
    from dateutil.parser import parse
    date = datetime.strptime(date, '%Y%m%d%H')
    delta = date - datetime(2000,1,1,0,0,0)
    hourssince = delta.total_seconds() / 3600
    return int(hourssince)

def hurricane_track(hurrfile):
    import csv
    with open(hurrfile) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        date, lat, lon = [], [], []
        for i, row in enumerate(reader):
            if i > 0:
                lat.append(float(row[8]))
                date.append(hurrtimeconv(row[0]))
                if float(row[9])<0:
                    lon.append(360+float(row[9]))
                else: 
                    lon.append(float(row[9]))
        return date, lat, lon

def find_hycom_dir(hours):
    import netCDF4
    import numpy as np  
    hycom_url = 'http://tds.hycom.org/thredds/dodsC/GLBu0.08/'
    hycom_extensions = ['expt_91.2','expt_91.1', 'expt_91.0', 'expt_90.9']
    hurrtime = hours[0]
    for ext in hycom_extensions:
        url = hycom_url+ext
        data = netCDF4.Dataset(url)
        dtime = data.variables['time'][:]
        if dtime[0] < hurrtime and dtime[-1] > hurrtime:
            return url

@jit
def find_time_index(date,hour):
	dsize = date.shape[0]
	min_d = None
	index  = None
	for i in range(dsize):
		dt = date[i]
		d = (dt-float(hour))**2
		if min_d == None or d < min_d:
			min_d = d
			index = i
	return index

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

hurrfile = "/Users/cew145/Desktop/RUCOOL_DAP/Hurricanefiles/al092016_track.csv"

hours, hurrlat, hurrlon = hurricane_track(hurrfile)

start = -30

hycom_url = find_hycom_dir(hours)
locations = zip(hurrlat,hurrlon,hours)
#locations = locations[start:]

print hycom_url
data = netCDF4.Dataset(hycom_url)
dtime = data.variables['time'][:]
temperatures = data.variables['water_temp']
#print temperatures.dimensions


#hour_ind = find_time_index(dtime,hours[0])
first = hours[0]


d_list = []
t_list = []
T_list = []
for loc in locations:
    here = [loc[0],loc[1]]
    when = loc[2]
    dt = when - first
    when_ind = find_time_index(dtime,when)
    T, d = hycomScrubber(hycom_url, 'water_temp', here, when_ind)
    t = [dt for i in d]
    #print here, ymdh(when)
    #plt.plot(d[:10],T[:10])
    d_list.append(d)
    t_list.append(t)
    T_list.append(T)
    #temp_matrix.append(zip(T,d,t))


#temp_matrix = np.asarray(temp_matrix)
from itertools import chain
time = list(chain.from_iterable(t_list))
depth = list(chain.from_iterable(d_list))
Temps = list(chain.from_iterable(T_list))

x = time
y = depth
z = Temps
xyz = zip(x,y,z)

print len(xyz)


xyz = [point if isinstance(point[2],np.float32) else (point[0], point[1], -1) for point in xyz]

#xyz = [point for point in xyz if isinstance(point[2],np.float32)]
x = [point[0] for point in xyz]
y = [-1*point[1] for point in xyz]
z = [point[2] for point in xyz]
print len(y)


fig = plt.figure()
ax = fig.add_subplot(211) #plt.subplots(1,2, sharex=True, sharey=True)
ax.set_title("Temperature Profile")
plt.xlabel('Date') # x-axis label
plt.ylabel('Depth (m)') # y-axis label
import cmocean
plt.tricontour(x, y, z, 20, linewidths=0.5, colors='k')
plt.tricontourf(x,y,z, 20, cmap = cmocean.cm.thermal) 
plt.scatter(x,y, edgecolors='none', c=z, cmap = cmocean.cm.thermal)
plt.clim(5,31)
plt.axis([min(x), max(x), min(y), max(y)])
plt.colorbar()
#ax[1].plot(x,y, 'ko ')
#ax[0].plot(x,y, 'ko ')
#plt.subplot(2, 1, 2)



from mpl_toolkits.basemap import Basemap, cm
ax = fig.add_subplot(212)
ax.set_title("Hurricane Path")
m = Basemap(llcrnrlon=min(hurrlon)-1,llcrnrlat=min(hurrlat)-1,urcrnrlon=max(hurrlon)+1, \
    urcrnrlat=max(hurrlat)+1)
#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='coral',lake_color='aqua')
m.drawcoastlines()
m.plot(hurrlon,hurrlat,color='blue')
#m.scatter(hurrlon[start:],hurrlat[start:],color='red')
m.scatter(hurrlon,hurrlat,color='red')

plt.savefig('fig_test.png')
plt.show()


"""
deptim = izip(d_list,t_list,T_list)
for dt in deptim:
    plt.scatter(dt[1], dt[0])
plt.show()"""








