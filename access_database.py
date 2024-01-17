import os
import os.path
import urllib.request

import numpy as np
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
                    SH.SnapNum = 27 \
                    and  SH.GroupNumber = %d \
                    and SH.SubGroupNumber = %d' % (group_number, subgroup_number)

    # Connect to database and execute the query #
    connnect = sql.connect('', password='')
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
    # Find the specific galaxy in the database and extract its ids #
    query = 'SELECT \
        REF.Redshift as z, \
        REF.SnapNum as snap, \
        REF.GalaxyID as galaxy, \
        REF.TopLeafID as top_leaf, \
        REF.DescendantID as descendant, \
        REF.LastProgID as last_progenitor \
    FROM \
        RefL0100N1504_SubHalo as REF \
    WHERE \
        REF.SnapNum = 27 \
        and REF.GroupNumber = %d \
        and REF.SubGroupNumber = %d' % (group_number, subgroup_number)

    # Connect to database and execute the query #
    connnect = sql.connect('', password='')
    sql_data = sql.execute_query(connnect, query)

    # Save the result in a data frame #
    df = pd.DataFrame(sql_data, columns=['z', 'snap', 'galaxy', 'top_leaf', 'descendant', 'last_progenitor'], index=[0])
    main_galaxy = df['galaxy'][0]
    main_descendant = df['descendant'][0]

    # Navigate through all branches and get all the progenitors #
    query = 'SELECT \
        SH.Redshift as z, \
        SH.SnapNum as snap, \
        SH.GalaxyID as galaxy, \
        SH.DescendantID as descendant, \
        SH.LastProgID as last_progenitor, \
        SH.MassType_Star as stellar_mass \
    FROM \
        RefL0100N1504_Subhalo as SH, \
        RefL0100N1504_Subhalo as REF, \
        RefL0100N1504_Aperture as AP \
    WHERE \
        Ref.SnapNum=27 \
        and REF.GalaxyID = %d \
        and AP.ApertureSize = 30 \
        and SH.MassType_Star >= 1E7 \
        and AP.GalaxyID = REF.GalaxyID \
        and ((SH.SnapNum > REF.SnapNum and REF.GalaxyID between SH.GalaxyID and SH.TopLeafID) or (SH.SnapNum >= 25 and SH.GalaxyID between ' \
            'REF.GalaxyID and REF.LastProgID)) \
    ORDER BY \
        SH.Redshift' % main_galaxy

    # Connect to database and execute the query #
    connnect = sql.connect("", password="")
    sql_data = sql.execute_query(connnect, query)

    # Save the result in a data frame #
    df = pd.DataFrame(sql_data, columns=['z', 'snap', 'galaxy', 'descendant', 'last_progenitor', 'stellar_mass'])

    # Find how many progenitors exist in the previous snap (26) to check if the galaxy had a merger #
    n_mergers, n_minor_mergers, n_major_mergers = 0, 0, 0
    if len(df.loc[lambda df:df['snap'] == 26, :]) == 1:
        merger_flag = 0
    # If it had merger(s) check if it(they) was(were) minor or major based on the mass ratios (0.3 threshold) of the involved galaxies #
    elif len(df.loc[lambda df:df['snap'] == 26, :]) > 1:
        df_26 = df.loc[lambda df:df['snap'] == 26, 'stellar_mass'].to_numpy()
        ratios = [x / max(df_26) for x in df_26]
        n_mergers = len(ratios) - 1
        if 0 <= np.sum(ratios) < 1 + len(ratios) * 0.3:  # Minor merger(s).
            merger_flag = 1
            n_minor_mergers = n_mergers
        elif 1 + len(ratios) * 0.3 <= np.sum(ratios) < 1 + len(ratios) * 1:  # Major merger(s).
            merger_flag = 2
            n_major_mergers = n_mergers
        else:
            raise KeyError("wrong merger ratio in create_merger_tree")
    else:
        raise KeyError("wrong number of mergers in create_merger_tree")

    return df, merger_flag, n_mergers, n_minor_mergers, n_major_mergers
