import os
import urllib.request

import pandas as pd
import eagleSqlTools as sql

simulation = 'RefL0100N1504'
data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.

query = "SELECT \
                SH.Image_Face as face, \
                SH.Image_Edge as edge \
            FROM \
                RefL0100N1504_SubHalo as SH \
            WHERE \
                SH.SnapNum = 27 and  \
                SH.SubGroupNumber = 0 and \
                SH.GroupNumber = 1"

# Connect to database and execute the query #
connnect = sql.connect("clovell", password="eKD5W17A")
myData = sql.execute_query(connnect, query)

# Save the result in a data frame #
df = pd.DataFrame(myData, columns=['face', 'edge'], index=[0])

# Remove the unnecessary characters from the URL #
df['face'] = df['face'].str.replace('"<img src=', '').str.replace('>"', '').str.replace("'", '')
df['edge'] = df['edge'].str.replace('"<img src=', '').str.replace('>"', '').str.replace("'", '')

# Extract the file name #
df = df.assign(file_name=lambda x: x.face)
df['file_name'] = df['file_name'].str.replace('http://virgodb.cosma.dur.ac.uk/eagle-webstorage/' + simulation + '_Subhalo/', '')

df = df.assign(filename=lambda x: simulation + '' + x.file_name)
df['image'] = df.apply(lambda x: download_image(x.face, x.filename, simulation), axis=1)
df.to_csv('EAGLEimagesevolutiondf' + simulation + '.csv')


def download_image(url, filename, simulation):
    urllib.request.urlretrieve(url, simulation + '/' + filename)
    local_path_image = os.path.join(data_path, simulation + '/' + filename)
    return (local_path_image)