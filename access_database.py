import os
import os.path
import urllib.request

import pandas as pd
import eagleSqlTools._eagleSqlTools as sql

plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/images/'  # Path to save data.
url = 'http://virgodb.cosma.dur.ac.uk/eagle-webstorage/RefL0100N1504_Subhalo/'  # URL from the database.
if not os.path.exists(plots_path):
    os.makedirs(plots_path)


def download_image(group_number, subgroup_number):
    """
    Download mock gri face-on and edge-on images from the EAGLE database for a given galaxy.
    :param group_number: from read_add_attributes.py.
    :param subgroup_number: from read_add_attributes.py.
    :return:
    """
    query = 'SELECT \
                SH.Image_Face as face, \
                SH.Image_Edge as edge \
                FROM \
                    RefL0100N1504_SubHalo as SH \
                WHERE \
                    SH.SnapNum = 27 and  \
                    SH.GroupNumber = %d and \
                    SH.SubGroupNumber = %d' % (group_number, subgroup_number)
    
    # Connect to database and execute the query #
    connnect = sql.connect("hnz327", password="HRC478zd")
    sql_data = sql.execute_query(connnect, query)
    
    # Save the result in a data frame and remove the unnecessary characters #
    df = pd.DataFrame(sql_data, columns=['face', 'edge'], index=[0])
    df['face'] = df['face'].str.replace('"<img src=', '').str.replace('>"', '').str.replace("'", '').str.replace(url, '')
    df['edge'] = df['edge'].str.replace('"<img src=', '').str.replace('>"', '').str.replace("'", '').str.replace(url, '')
    galaxy_id = df['face'].item().split('_')[1]
    
    # Check if the images exist, if not download and save them #
    if not os.path.isfile(plots_path + df['face'].item()):
        urllib.request.urlretrieve(url + df['face'].item(), plots_path + df['face'].item())
    if not os.path.isfile(plots_path + df['edge'].item()):
        urllib.request.urlretrieve(url + df['edge'].item(), plots_path + df['edge'].item())
    
    return galaxy_id


def create_merger_tree(group_number, subgroup_number):
    """
    Create a merger tree for a given galaxy.
    :param group_number: from read_add_attributes.py.
    :param subgroup_number: from read_add_attributes.py.
    :return:
    """
    # Find the specific galaxy in the database and extract its ids. #
    for redshift in range(27, 25, -1):
        print(redshift)
        if redshift == 27:
            query = 'SELECT \
                SH.GalaxyID as gid, \
                SH.TopLeafID as tlid, \
                SH.LastProgID as lpid, \
                SH.DescendantID as did \
            FROM \
                RefL0100N1504_SubHalo as SH \
            WHERE \
                SH.SnapNum = %d and  \
                SH.GroupNumber = %d and \
                SH.SubGroupNumber = %d' % (redshift, group_number, subgroup_number)
            
            # Connect to database and execute the query #
            connnect = sql.connect("hnz327", password="HRC478zd")
            sql_data = sql.execute_query(connnect, query)
            
            # Save the result in a data frame #
            df = pd.DataFrame(sql_data, columns=['gid', 'tlid', 'lpid', 'did'], index=[0])
            gid = df['gid'][0]
            tlid = df['lpid'][0]
            dfs = [df]
            print(df)
        else:
            # Navigate through the main branch and get all the progenitors. #
            print(df)
            query = 'SELECT \
                DES.GalaxyID as gid, \
                DES.TopLeafID as tlid, \
                DES.LastProgID as lpid, \
                DES.DescendantID as did \
            FROM \
                RefL0100N1504_SubHalo as DES, \
                RefL0100N1504_Subhalo as PROG \
            WHERE \
                PROG.SnapNum = %d and \
                PROG.MassType_Star between 1E9 and 1E12 and \
                PROG.GalaxyID between DES.GalaxyID and DES.LastProgID' % (redshift)
            # Connect to database and execute the query #
            connnect = sql.connect("hnz327", password="HRC478zd")
            sql_data = sql.execute_query(connnect, query)
            df = pd.DataFrame(sql_data, columns=['gid', 'tlid', 'lpid', 'did'])
            dfs.append(df)
            tree = pd.concat(dfs, ignore_index=True)
    print(tree)
    return tree
