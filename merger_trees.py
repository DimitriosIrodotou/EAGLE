import os
import re
import time
import warnings
import matplotlib
import access_database

matplotlib.use('Agg')

import numpy as np
import networkx as nx
import matplotlib.cbook
import matplotlib.pyplot as plt

from astropy.cosmology import Planck13

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
        
        for group_number in range(19, 26):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):  # Get centrals only.
                start_local_time = time.time()  # Start the local time.
                
                self.plot(group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MergerTree for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, group_number, subgroup_number):
        """
        Plot a merger tree.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        
        # Get the merger tree information from the database and display it in a hierarchical tree structure #
        df, flag, n_mergers = access_database.create_merger_tree(group_number, subgroup_number)
        G = nx.from_pandas_edgelist(df=df, source='galaxy', target='descendant', create_using=nx.Graph)
        G.add_nodes_from(nodes_for_adding=df['galaxy'].tolist())
        tree = nx.bfs_tree(G, df['galaxy'][0])
        
        # Create the hierarchical tree structure and display it #
        # pos = nx.drawing.nx_agraph.graphviz_layout(tree, root=df['galaxy'][0])
        pos = self.hierarchy_pos(tree, root=df['galaxy'][0])
        
        # Reorder df to assign the colors to each node based on lookback time and plot #
        df['lbt'] = df.apply(lambda x:round(Planck13.lookback_time(x.z).value, 1), axis=1)
        df = df.set_index('galaxy')
        df = df.reindex(tree.nodes())
        nx.draw_networkx(tree, pos=pos, with_labels=True, node_color=np.log10(df['stellar_mass']), cmap='jet', alpha=0.7,
                         vmin=np.log10(df['stellar_mass'].min()), vmax=np.log10(df['stellar_mass'].max()), ax=axis)
        
        # Create a mappable for the colorbar #
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=np.log10(df['stellar_mass'].min()), vmax=np.log10(df['stellar_mass'].max())))
        sm.set_array([])
        plt.colorbar(sm).set_label(r'$\mathrm{M_{\bigstar}/M_{\odot}}$')
        
        axis.set_ylabel(r'$\mathrm{z}$')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.tick_params(left=True, labelleft=True)
        plt.text(0.1, 1, r'$\mathrm{Flag: }$' + str(flag), fontsize=12, transform=axis.transAxes)
        axis.set_yticklabels(('', '0.27', '', '0.18', '', '0.1', '', '0.0', ''))
        
        # Save the figure #
        plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'MT' + '-' + date + '.png', bbox_inches='tight')
        return None
    
    
    @staticmethod
    def hierarchy_pos(G, root, xcenter=0.5, width=1.0, vert_gap=0.2, vert_loc=0):
        """
        Return the positions to plot a merger tree in a hierarchical layout where a leaf node at a higher level gets the entire space allocated to
        its descendant leaves.

        :param G: the graph (must be a tree).
        :param root: the root node of the tree.
        :param xcenter: the horizontal location of root.
        :param width: the horizontal space allocated for a branch.
        :param vert_gap: the gap between levels of hierarchy.
        :param vert_loc: the vertical location of root
        :return: pos: a dictionary with nodes as keys and positions as values
        """
        
        
        def _hierarchy_pos(G, root, leftmost, width, leafdx=0.2, vert_gap=vert_gap, vert_loc=0.8, xcenter=xcenter, rootpos=None, leafpos=None,
                           parent=None):
            """
            :param G: the graph (must be a tree).
            :param root: the root node of the tree.
            :param leftmost:
            :param width: the horizontal space allocated for a branch.
            :param leafdx:
            :param vert_gap: the gap between levels of hierarchy.
            :param vert_loc: the vertical location of root
            :param xcenter: the horizontal location of root.
            :param rootpos: the position of the root.
            :param leafpos: the position of the leaf.
            :param parent: parent of this branch.
            :return:
            """
            
            if rootpos is None:
                rootpos = {root:(xcenter, vert_loc)}
            else:
                rootpos[root] = (xcenter, vert_loc)
            if leafpos is None:
                leafpos = {}
            
            children = list(G.neighbors(root))
            leaf_count = 0
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                rootdx = width / len(children)
                nextx = xcenter - width / 2 - rootdx / 2
                for child in children:
                    nextx += rootdx
                    rootpos, leafpos, newleaves = _hierarchy_pos(G, child, leftmost + leaf_count * leafdx, width=rootdx, leafdx=leafdx,
                                                                 vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx, rootpos=rootpos,
                                                                 leafpos=leafpos, parent=root)
                    leaf_count += newleaves
                
                leftmostchild = min((x for x, y in [leafpos[child] for child in children]))
                rightmostchild = max((x for x, y in [leafpos[child] for child in children]))
                leafpos[root] = ((leftmostchild + rightmostchild) / 2, vert_loc)
            else:
                leaf_count = 1
                leafpos[root] = (leftmost, vert_loc)
            return rootpos, leafpos, leaf_count
        
        
        xcenter = width / 2.
        if isinstance(G, nx.DiGraph):
            leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node) == 0])
        elif isinstance(G, nx.Graph):
            leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node) == 1 and node != root])
        rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, leafdx=width * 1. / leafcount, vert_gap=vert_gap, vert_loc=vert_loc,
                                                      xcenter=xcenter)
        pos = {}
        for node in rootpos:
            leaf_vs_root_factor = 1
            pos[node] = (leaf_vs_root_factor * leafpos[node][0] + (1 - leaf_vs_root_factor) * rootpos[node][0], leafpos[node][1])
        xmax = max(x for x, y in pos.values())
        for node in pos:
            pos[node] = (pos[node][0] * width / xmax, pos[node][1])
        return pos


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/MT/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = MergerTree(simulation_path, tag)
