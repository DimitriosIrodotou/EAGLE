import os.path
import urllib.request

import pandas as pd
import eagleSqlTools as sql

data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save data.
url = 'http://virgodb.cosma.dur.ac.uk/eagle-webstorage/RefL0100N1504_Subhalo/'  # URL from the database.


def download_image(group_number, subgroup_number):
    query = "SELECT \
                    SH.Image_Face as face, \
                    SH.Image_Edge as edge \
                FROM \
                    RefL0100N1504_SubHalo as SH \
                WHERE \
                    SH.SnapNum = 27 and  \
                    SH.GroupNumber = %d and \
                    SH.SubGroupNumber = %d" % (group_number, subgroup_number)
    
    # Connect to database and execute the query #
    connnect = sql.connect("hnz327", password="HRC478zd")
    myData = sql.execute_query(connnect, query)
    
    # Save the result in a data frame and remove the unnecessary characters #
    df = pd.DataFrame(myData, columns=['face', 'edge'], index=[0])
    df['face'] = df['face'].str.replace('"<img src=', '').str.replace('>"', '').str.replace("'", '').str.replace(url, '')
    df['edge'] = df['edge'].str.replace('"<img src=', '').str.replace('>"', '').str.replace("'", '').str.replace(url, '')
    galaxy_id = df['face'].item().split('_')[1]
    
    # Check if the images exist, if not download and save them #
    if not os.path.isfile(data_path + 'images/' + df['face'].item()):
        urllib.request.urlretrieve(url + df['face'].item(), data_path + 'images/' + df['face'].item())
    if not os.path.isfile(data_path + 'images/' + df['edge'].item()):
        urllib.request.urlretrieve(url + df['edge'].item(), data_path + 'images/' + df['edge'].item())
    
    return galaxy_id