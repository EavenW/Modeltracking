from bs4 import BeautifulSoup
import zipfile
import os, sys, string
import pandas as pd

kmz_file = sys.argv[1]
fileloc = os.path.dirname(kmz_file)
tempfolder = '{}/tmp'.format(fileloc)
hurrname = string.split(os.path.basename(kmz_file),'_')[0]
name = hurrname+'_best_track.kmz'

print kmz_file

with zipfile.ZipFile(kmz_file,"r") as zip_ref:
    zip_ref.extractall(tempfolder)

file = '{}/{}.kml'.format(tempfolder, hurrname)


vars = ['atcfdtg', 'stormnum', 'stormname', 'basin', 'stormtype', 'intensity', \
     'intensitymph', 'intensitykph', 'lat', 'lon',  'minsealevelpres', 'dtg']

data = []

with open(file) as f:
    soup = BeautifulSoup(f, 'lxml')
    for placemark in soup.find_all('placemark'):
        line = []
        for var in vars:
            line.append(placemark.find(var).contents[0])
        line = tuple(line)
        data.append(line)

df = pd.DataFrame(data, columns=vars)
df.to_csv(os.path.join(fileloc, '{}_track.csv'.format(hurrname)), index=False)

# Clean
os.system('rm -rf %s' % tempfolder)