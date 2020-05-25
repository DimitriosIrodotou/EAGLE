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
    query = 'SELECT \
        SH.GalaxyID as galaxy, \
        SH.TopLeafID as top_leaf, \
        SH.LastProgID as last_progenitor, \
        SH.DescendantID as descendant \
    FROM \
        RefL0100N1504_SubHalo as SH \
    WHERE \
        SH.SnapNum = 27 and  \
        SH.GroupNumber = %d and \
        SH.SubGroupNumber = %d' % (group_number, subgroup_number)
    
    # Connect to database and execute the query #
    connnect = sql.connect("hnz327", password="HRC478zd")
    sql_data = sql.execute_query(connnect, query)
    
    # Save the result in a data frame #
    df = pd.DataFrame(sql_data, columns=['galaxy', 'top_leaf', 'last_progenitor', 'descendant'], index=[0])
    main_galaxy = df['galaxy'][0]
    main_top_leaf = df['last_progenitor'][0]
    dfs = [df]
    
    # Navigate through the main branch and get all the progenitors. #
    query = 'SELECT \
        PROG.GalaxyID as galaxy, \
        PROG.TopLeafID as top_leaf, \
        PROG.LastProgID as last_progenitor, \
        PROG.DescendantID as descendant \
    FROM \
        RefL0100N1504_Subhalo as PROG \
    WHERE \
        PROG.SnapNum > 10 and \
        PROG.MassType_Star >= 1E5 and \
        PROG.GalaxyID between %d and %d' % (main_galaxy, main_top_leaf)
    
    # Connect to database and execute the query #
    connnect = sql.connect("hnz327", password="HRC478zd")
    sql_data = sql.execute_query(connnect, query)
    
    # Save the result in a data frame #
    df = pd.DataFrame(sql_data, columns=['galaxy', 'top_leaf', 'last_progenitor', 'descendant'])
    dfs.append(df)
    tree = pd.concat(dfs, ignore_index=True)
    print(tree)
    
    return tree
