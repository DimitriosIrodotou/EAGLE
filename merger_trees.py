import os
import re
import time
import random
import warnings
import matplotlib
import access_database
matplotlib.use('Agg')

import numpy as np
import networkx as nx
import matplotlib.cbook
import matplotlib.pyplot as plt

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class MergerTree:
    """
    For each galaxy create: a merger tree.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        
        for group_number in range(25, 26):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):  # Get centrals only.
                start_local_time = time.time()  # Start the local time.
                
                # Load data from numpy arrays #
                stellar_data_tmp = np.load(
                    data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy', allow_pickle=True)
                stellar_data_tmp = stellar_data_tmp.item()
                print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MergerTree for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(stellar_data_tmp, group_number, subgroup_number):
        """
        Plot a merger tree
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: None
        """
        # Generate the figure and define its parameters #
        # plt.close()
        # plt.figure(0, figsize=(16, 9))
        
        # Plot the 2D surface density projections #
        
        df2 = access_database.create_merger_tree(group_number, subgroup_number)
        
        # galaxyid = df2['gid'][0]
        #
        # fig, ax = plt.subplots()
        # G = nx.from_pandas_edgelist(df=df2, source='DescGalaxyID', target='DescID', create_using=nx.Graph)
        # G.add_nodes_from(nodes_for_adding=df2.DescGalaxyID.tolist())
        # # df2=df2.reindex(G.nodes())
        # tree = nx.bfs_tree(G, galaxyid)
        # zlist = df2.lbt.tolist()
        # colourparam = 'red'
        # # pos = hierarchy_pos(tree, df2, root=galaxyid)
        # print("EEEEE")
        # nx.draw_networkx(G, with_labels=False, font_size=9, node_size=50, node_color=df2[colourparam], cmap=plt.cm.plasma,
        #                  vmin=df2[colourparam].min(), vmax=df2[colourparam].max(), ax=ax)
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=df2[colourparam].min(), vmax=df2[colourparam].max()))
        # sm.set_array([])
        #
        # ax.tick_params(left=True, labelleft=True)
        # locs, labels = plt.yticks()
        # print('locs={}, labels={}'.format(locs, labels))
        # print(df2.z.min())
        # # labels2 = np.linspace(-df2.lbt.max(), -df2.lbt.min(), len(locs))
        # # labels2=np.around(labels2,decimals=1)
        # labels2 = np.array(locs) * (-1)
        # labels2.sort()
        # print(labels2)
        # plt.yticks(locs, labels2)
        # plt.ylabel('z')
        # cbar = plt.colorbar(sm).set_label(colourparam)
        # plt.title('Galaxy Merger Tree for galaxy' + str(galaxyid))
        #
        # plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'MT' + '-' + date + '.png', bbox_inches='tight')
        return None

    #
    # def hierarchy_pos(G, df, root=None, width=1., vert_gap=0.2, vert_loc=0, leaf_vs_root_factor=1):
    #     """
    #     If the graph is a tree this will return the positions to plot this in a
    #     hierarchical layout.
    #
    #     Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    #     but with some modifications.
    #
    #     We include this because it may be useful for plotting transmission trees,
    #     and there is currently no networkx equivalent (though it may be coming soon).
    #
    #     There are two basic approaches we think of to allocate the horizontal
    #     location of a node.
    #
    #     - Top down: we allocate horizontal space to a node.  Then its ``k``
    #       descendants split up that horizontal space equally.  This tends to result
    #       in overlapping nodes when some have many descendants.
    #     - Bottom up: we allocate horizontal space to each leaf node.  A node at a
    #       higher level gets the entire space allocated to its descendant leaves.
    #       Based on this, leaf nodes at higher levels get the same space as leaf
    #       nodes very deep in the tree.
    #
    #     We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    #     determining how much of the horizontal space is based on the bottom up
    #     or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    #     down.
    #
    #
    #     **root** the root node of the tree
    #     - if the tree is directed and this is not given, the root will be found and used
    #     - if the tree is directed and this is given, then the positions will be
    #       just for the descendants of this node.
    #     - if the tree is undirected and not given, then a random choice will be used.
    #
    #     **width** horizontal space allocated for this branch - avoids overlap with other branches
    #
    #     **vert_gap** gap between levels of hierarchy
    #
    #     **vert_loc** vertical location of root
    #
    #     **leaf_vs_root_factor**
    #
    #     xcenter: horizontal location of root
    #     :param G: the graph
    #     :param df:
    #     :param root:
    #     :param width:
    #     :param vert_gap:
    #     :param vert_loc:
    #     :param leaf_vs_root_factor:
    #     :return:
    #     """
    #     if not nx.is_tree(G):
    #         raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
    #
    #     if root is None:
    #         if isinstance(G, nx.DiGraph):
    #             root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
    #         else:
    #             root = random.choice(list(G.nodes))
    #
    #
    #     def _hierarchy_pos(G, root, leftmost, width, leafdx=0.2, vert_gap=0.2, vert_loc=0.8, xcenter=0.5, rootpos=None, leafpos=None, parent=None):
    #         '''
    #         see hierarchy_pos docstring for most arguments
    #
    #         pos: a dict saying where all nodes go if they have been assigned
    #         parent: parent of this branch. - only affects it if non-directed
    #
    #         '''
    #
    #         if rootpos is None:
    #             rootpos = {root:(xcenter, vert_loc)}
    #         else:
    #             rootpos[root] = (xcenter, vert_loc)
    #         if leafpos is None:
    #             leafpos = {}
    #         children = list(G.neighbors(root))
    #         leaf_count = 0
    #         if not isinstance(G, nx.DiGraph) and parent is not None:
    #             children.remove(parent)
    #         if len(children) != 0:
    #             rootdx = width / len(children)
    #             nextx = xcenter - width / 2 - rootdx / 2
    #             for child in children:
    #                 nextx += rootdx
    #                 rootpos, leafpos, newleaves = _hierarchy_pos(G, child, leftmost + leaf_count * leafdx, width=rootdx, leafdx=leafdx, vert_gap=vert_gap,
    #                                                              vert_loc=vert_loc - vert_gap, xcenter=nextx, rootpos=rootpos, leafpos=leafpos,
    #                                                              parent=root)
    #                 leaf_count += newleaves
    #
    #             leftmostchild = min((x for x, y in [leafpos[child] for child in children]))
    #             rightmostchild = max((x for x, y in [leafpos[child] for child in children]))
    #             leafpos[root] = ((leftmostchild + rightmostchild) / 2, vert_loc)
    #         else:
    #             leaf_count = 1
    #             leafpos[root] = (leftmost, vert_loc)
    #         # pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
    #         # print(leaf_count)
    #         return rootpos, leafpos, leaf_count
    #
    #
    #     xcenter = width / 2.
    #     if isinstance(G, nx.DiGraph):
    #         leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node) == 0])
    #     elif isinstance(G, nx.Graph):
    #         leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node) == 1 and node != root])
    #     rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, leafdx=width * 1. / leafcount, vert_gap=vert_gap, vert_loc=vert_loc,
    #                                                   xcenter=xcenter)
    #     pos = {}
    #     for node in rootpos:
    #         pos[node] = (leaf_vs_root_factor * leafpos[node][0] + (1 - leaf_vs_root_factor) * rootpos[node][0], leafpos[node][
    #             1])  # pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for
    #         # node in rootpos}
    #     xmax = max(x for x, y in pos.values())
    #     for node in pos:
    #         pos[node] = (pos[node][0] * width / xmax, df.loc[df['DescGalaxyID'] == node]['lbt'].item())
    #     return pos


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/MT/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = MergerTree(simulation_path, tag)
