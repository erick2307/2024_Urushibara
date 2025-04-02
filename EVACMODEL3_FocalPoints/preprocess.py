import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import folium
from folium import plugins
import os
import warnings
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import shortest_path
warnings.simplefilter(action='ignore',category=FutureWarning) 
warnings.simplefilter(action='ignore',category=RuntimeWarning) 
warnings.simplefilter(action='ignore',category=DeprecationWarning)

def check_folders(foldername='Input', sp=False):
    """
    Checks if a folder exists, renames it if it does, and creates necessary subdirectories.
    This function performs the following steps:
    1. Checks if a directory with the specified name exists.
    2. If the directory exists, it attempts to rename it by appending '_CAREFUL_PREVIOUS_INPUT' to the name.
       - If renaming fails due to an OSError, it prints an error message and raises the exception.
    3. Creates a new directory with the specified name.
    4. Creates a subdirectory named 'StatesMatrices' within the newly created directory.
    Args:
        foldername (str): The name of the folder to check and create. Defaults to 'Input'.
    Raises:
        OSError: If renaming the existing directory fails.
    """

    if os.path.isdir(foldername):
        try:
            os.rename(foldername,f'{foldername}_CAREFUL_PREVIOUS_INPUT')
        except OSError:
            print('TOO MANY CASES WITH THE SAME NAME')
            raise
    os.makedirs(foldername, exist_ok=True)
    os.makedirs(f'./{foldername}/Figures', exist_ok=True)
    os.makedirs(f'./{foldername}/Graph', exist_ok=True)
    if not sp:
        os.makedirs(f'./{foldername}/Figures_BestChoices', exist_ok=True)
        os.makedirs(f'./{foldername}/Simulations', exist_ok=True)
        os.makedirs(f'./{foldername}/StatesMatrices', exist_ok=True)
    print(f'Folder {foldername} created')

def download_nwk(polygon, show=True, close=False, save=True, foldername='Input'):
    #get a graph
    G = ox.graph_from_polygon(polygon, network_type='all', simplify=True)
    # Get node positions
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    # Plot
    fig, ax = ox.plot_graph(G,node_color='blue', bgcolor='white', show=show, close=close, save=save, filepath=f'./{foldername}/Figures/nwk_simple.png')
    # Add custom node labels (here using node IDs)
    for node, (x, y) in pos.items():
        ax.text(x, y, str(node), fontsize=8, color='red')
    filepath=f'./{foldername}/Figures/nwk_nodes.png'
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    #save graph as geojson files
    nodes, edges = ox.graph_to_gdfs(G)
    # Save edges as GeoJSON (recommended for road network)
    edges.to_file(f'./{foldername}/Graph/Gedges.geojson', driver='GeoJSON')
    # Optionally save nodes too
    nodes.to_file(f'./{foldername}/Graph/Gnodes.geojson', driver='GeoJSON')
    G_proj = ox.project_graph(G)
    G_und = G_proj.to_undirected()
    print('Graph downloaded')
    return G_und


def download_point_nwk(lat=38.435220,lon=141.303816,radius=1000, show=True, close=False, save=True, foldername='Input'):
    """
    Downloads a network graph from OpenStreetMap based on a specified location and radius, 
    and optionally plots and saves the graph.

    Parameters:
    lat (float): Latitude of the center point. Default is 38.435220.
    lon (float): Longitude of the center point. Default is 141.303816.
    radius (int): Radius in meters to define the area for the network graph. Default is 1000.
    show (bool): Whether to display the plot. Default is True.
    save (bool): Whether to save the plot as an image file. Default is True.
    foldername (str): Name of the folder to save the plot image. Default is 'Input'.

    Returns:
    networkx.Graph: An undirected, projected graph of the specified area.
    """
    #get a graph
    G = ox.graph_from_point(center_point=(lat,lon), network_type='all', dist=radius, simplify=True)
    # Get node positions
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
    # Plot
    fig, ax = ox.plot_graph(G,node_color='blue', bgcolor='white', show=show, close=close, save=save, filepath=f'./{foldername}/Figures/nwk_simple.png')
    # Add custom node labels (here using node IDs)
    for node, (x, y) in pos.items():
        ax.text(x, y, str(node), fontsize=8, color='red')
    filepath=f'./{foldername}/Figures/nwk_nodes.png'
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    #save graph as geojson files
    nodes, edges = ox.graph_to_gdfs(G)
    # Save edges as GeoJSON (recommended for road network)
    edges.to_file(f'./{foldername}/Graph/Gedges.geojson', driver='GeoJSON')
    # Optionally save nodes too
    nodes.to_file(f'./{foldername}/Graph/Gnodes.geojson', driver='GeoJSON')
    G_proj = ox.project_graph(G)
    G_und = G_proj.to_undirected()
    print('Graph downloaded')
    return G_und

def unpack_columns(df, c):
    """
    Unpacks dictionaries stored in a specified column of a DataFrame into separate columns.
    Args:
        df (pandas.DataFrame): The DataFrame containing the column with dictionaries.
        c (str): The name of the column containing dictionaries to unpack.
    Returns:
        pandas.DataFrame: The DataFrame with new columns added, each corresponding to a key in the dictionaries.
    """
    keys = []
    for row in df.iterrows():
        keys.append(row[1][c].keys())
    keys = list(set([item for sublist in keys for item in sublist]))

    for key in keys:
        df[key] = df[c].apply(lambda x: x.get(key))
    return df

def get_nodedb(G,evacnodes=[0],foldername='Input'):
    """
    Generates a node database from a graph and saves it as a CSV file.
    Parameters:
    G (networkx.Graph): The input graph from which to generate the node database.
    evacnodes (list, optional): List of node IDs that are evacuation nodes. Default is [0].
    foldername (str, optional): The name of the folder where the CSV file will be saved. Default is 'Input'.
    Returns:
    pandas.DataFrame: A DataFrame containing the node database with columns:
                        '#number', 'Coord_x', 'Coord_y', 'evacuation', 'reward'.
    """
    Gdfn = pd.DataFrame(G.nodes(data=True))
    Gdfn = unpack_columns(Gdfn,1)      
    Gdfn.drop([1,'street_count','highway', 'lat', 'lon'],axis=1,inplace=True)
    Gdfn.rename(columns={0:'osmid', 'x':'Coord_x', 'y': 'Coord_y'},inplace=True)
    Gdfn['evacuation'] = 0
    Gdfn['reward'] = 1
    for node in evacnodes:
        Gdfn.loc[Gdfn['osmid'] == node,'evacuation'] = 1
        Gdfn.loc[Gdfn['osmid'] == node,'reward'] = 1000
        print(f'Node {node} is an evacuation node')
    Gdfn['#number'] = Gdfn.index
    Gdfn.index = Gdfn['osmid']
    Gdfn = Gdfn[['#number', 'Coord_x','Coord_y','evacuation','reward']]
    Gdfn.to_csv(f'./{foldername}/nodes0.csv',index=False)
    return Gdfn

def get_edgedb(G, Gdfn, foldername='Input'):
    """
    Processes the edges of a graph and saves the resulting DataFrame to a CSV file.
    Parameters:
    G (networkx.Graph): The input graph.
    Gdfn (pandas.DataFrame): The node database DataFrame.
    foldername (str): The name of the folder where the CSV file will be saved. Default is 'Input'.
    Returns:
    pandas.DataFrame: A DataFrame containing the processed edges of the graph with columns:
        - '#number': The index of the edge.
        - 'Node1': The starting node of the edge.
        - 'Node2': The ending node of the edge.
        - 'Length': The length of the edge.
        - 'Width': The width of the edge (default is 3).
    """
    Gdfe = pd.DataFrame(G.edges(data=True))
    Gdfe = unpack_columns(Gdfe,2)
    for row in Gdfe.iterrows():
        #erase loop edges
        if row[1][0] == row[1][1]:# or row[1][0] in row[1][2]['osmid'] or row[1][1] in row[1][2]['osmid']:
            Gdfe.drop(row[0],inplace=True)

        #erase edges with length 0
        if int(row[1][2]['length']) == 0:
            Gdfe.drop(row[0],inplace=True)
        
        #replace list elements with first element
        Gdfe['osmid'] = Gdfe['osmid'].apply(lambda x: x[0] if type(x) is list else x)
   
    Gdfe.reset_index(drop=True,inplace=True)
    Gdfe['#number'] = Gdfe.index.astype(np.int64)
    Gdfe['Node1'] = Gdfe[0].map(Gdfn['#number']).astype(np.int64)
    Gdfe['Node2'] = Gdfe[1].map(Gdfn['#number']).astype(np.int64)
    Gdfe['Length'] = Gdfe['length'].astype(np.int64)
    Gdfe['Width'] = 3
    Gdfe = Gdfe[['#number', 'Node1', 'Node2', 'Length', 'Width']]
    Gdfe.to_csv(f'./{foldername}/edges0.csv',index=False)
    return Gdfe

def get_fake_population(pop,foldername='Input'):
    # df = pd.read_csv(f'./{foldername}/nodes.csv')
    dfsp = pd.read_csv(f'./{foldername}/nextnode.csv', names=['#number', 'nextnode'])
    # noroute = dfsp[dfsp['nextnode']==-9999]['#number'].values
    yesroute = dfsp[dfsp['nextnode']!=-9999]['#number'].values 
    popdf = pd.DataFrame()
    popdf['#age']=np.zeros(pop) 
    popdf['Gender']=np.zeros(pop) 
    popdf['#Hh Type']=np.zeros(pop) 
    popdf['#Hh Id']=np.zeros(pop) 
    popdf['Node']=np.random.choice(yesroute,pop)
    popdf.sort_values(by='Node')
    # pop = pop[~pop['Node'].isin(noroute)]
    popdf.reset_index()
    popdf.to_csv(f'./{foldername}/population.csv', index=False)

def get_fake_population_times(pop, times=1,foldername='Input'):
    """
    Generates a fake population dataset based on the nodes information and saves it to a CSV file.

    Parameters:
    times (int): The multiplier for the number of times to replicate the nodes data. Default is 1.
    foldername (str): The name of the folder where the input nodes.csv file is located and where the output population.csv file will be saved. Default is 'Input'.

    Returns:
    None: The function saves the generated population data to a CSV file named 'population.csv' in the specified folder.
    """

    # df = pd.read_csv(f'./{foldername}/nodes.csv')
    dfsp = pd.read_csv(f'./{foldername}/nextnode.csv', names=['#number', 'nextnode'])
    # noroute = dfsp[dfsp['nextnode']==-9999]['#number'].values 
    yesroute = dfsp[dfsp['nextnode']!=-9999]['#number'].values 
    popdf = pd.DataFrame()
    popdf['#age']=np.zeros(pop * times) 
    popdf['Gender']=np.zeros(pop * times)
    popdf['Hh Type']=np.zeros(pop * times)
    popdf['Hh Id']=np.zeros(pop * times)
    popdf['Node']=np.random.choice(yesroute, pop * times) 
    popdf.sort_values(by='Node')
    popdf.reset_index()
    print(f'Population in file: {popdf.shape[0]}')
    popdf.to_csv(f'./{foldername}/population_{times}.csv', index=False)
    
def get_actionsdb_transitionsdb(foldername='Input'):
    """
    Generates and saves actions, transitions, probabilities, and rewards databases from input node and edge data.
    Parameters:
    foldername (str): The name of the folder containing the input CSV files 'nodes.csv' and 'edges.csv'. Default is 'Input'.
    The function performs the following steps:
    1. Loads node and edge data from CSV files located in the specified folder.
    2. Initializes actions, transitions, probabilities, and rewards databases with zeros.
    3. Iterates over each node to populate the databases based on the node and edge data.
    4. Saves the populated actions and transitions databases to CSV files in the specified folder.
    The input CSV files should have the following structure:
    - nodes.csv: Columns should include node ID and other relevant attributes.
    - edges.csv: Columns should include edge ID, source node ID, target node ID, and edge weight.
    The output CSV files will have the following structure:
    - actionsdb.csv: Columns include node ID, number of actions, and action details.
    - transitionsdb.csv: Columns include node ID, number of transitions, and transition details.
    """

    nodesdb = np.loadtxt(f"./{foldername}/nodes.csv", delimiter=',', skiprows=1)
    linksdb = np.loadtxt(f"./{foldername}/edges.csv", delimiter=',', skiprows=1)
    numNodes = nodesdb.shape[0]
    numLinks = linksdb.shape[0]
    
    actionsdb = np.zeros((numNodes, 12), dtype=np.int32)
    transitionsdb = np.zeros((numNodes, 12), dtype=np.int32)
    probdb = np.zeros((numNodes, 12), dtype=np.int32) 
    rewarddb = np.zeros((numNodes, 12), dtype=np.int32) 
    
    for i in range(numNodes):
        actionsdb[i,0] = nodesdb[i,0]
        transitionsdb[i,0] = nodesdb[i,0]
        rewarddb[i,0] = nodesdb[i,0]
        
        if nodesdb[i,3]:
            actionsdb[i,1] = 1
            actionsdb[i,2] = -1
            transitionsdb[i,1] = 1
            transitionsdb[i,2] = actionsdb[i,0]
            rewarddb[i,1] = 1
            rewarddb[i,2] = 0
            continue
            
        tmpLinksdb1 = linksdb[linksdb[:,1] == nodesdb[i,0]]
        tmpLinksdb2 = linksdb[linksdb[:,2] == nodesdb[i,0]]
        numlinks1 = tmpLinksdb1.shape[0]
        numlinks2 = tmpLinksdb2.shape[0]
        
        actionsdb[i,1] = numlinks1 + numlinks2
        transitionsdb[i,1] = numlinks1 + numlinks2
        rewarddb[i,1] = numlinks1 + numlinks2
        
        if numlinks1:
            actionsdb[i, 2: 2+numlinks1] = tmpLinksdb1[:,0]
            transitionsdb[i,2:2 + numlinks1] = tmpLinksdb1[:,2]
            rewarddb[i, 2:2+numlinks1] = -tmpLinksdb1[:,3]
        
        if numlinks2:
            actionsdb[i, 2+numlinks1 : 2+numlinks1+numlinks2] = tmpLinksdb2[:,0]
            transitionsdb[i, 2+numlinks1 : 2+numlinks1+numlinks2] = tmpLinksdb2[:,1]
            rewarddb[i, 2+numlinks1 : 2+numlinks1+numlinks2] = -tmpLinksdb2[:,3]

    ind = np.argmax(actionsdb[:,1])
    probdb[:,0:2] = actionsdb[:,0:2]
#    rewarddb[:,0:2] = actionsdb[:,0:2]
    
    for i in range(numNodes):
        numActions = actionsdb[i,1]
        if numActions:
            probdb[i,2:2+numActions] = np.ones(numActions)
    
    np.savetxt(f'./{foldername}/actionsdb.csv', actionsdb, delimiter=',', fmt='%d')
    np.savetxt(f'./{foldername}/transitionsdb.csv', transitionsdb, delimiter=',', fmt='%d')
    
    
def get_shortpath_file(foldername):
    """
    Generates a CSV file containing the next node in the shortest path to the nearest shelter for each node in a graph.
    Parameters:
    foldername (str): The name of the folder containing the 'nodes.csv' and 'edges.csv' files.
    The function performs the following steps:
    1. Reads node and edge data from CSV files located in the specified folder.
    2. Creates a graph using the NetworkX library and adds nodes and edges to it.
    3. Identifies shelter nodes (nodes with an evacuation value of 1).
    4. Calculates the shortest path lengths from each node to each shelter.
    5. Determines the closest shelter for each node.
    6. Calculates the next node in the shortest path to the closest shelter for each node.
    7. Saves the results to a CSV file named 'nextnode.csv' in the specified folder.
    The input CSV files should have the following structure:
    - nodes.csv: Columns ["#number", "Coord_x", "Coord_y", "Evacuation", "Reward"]
    - edges.csv: Columns ["#number", "Node1", "Node2", "Length", "Width"]
    The output CSV file 'nextnode.csv' will contain the following columns:
    - #number: The node number.
    - nextnode: The next node in the shortest path to the nearest shelter.
    """
    #create a graph
    G = nx.Graph()

    #read data
    Nodesdf = pd.read_csv(f'./{foldername}/nodes.csv',names=["#number","Coord_x","Coord_y","Evacuation","Reward"],skiprows=1)
    Edgesdf = pd.read_csv(f'./{foldername}/edges.csv',names=["#number","Node1","Node2","Length",'Width'],skiprows=1)

    #add nodes to graph
    for i,row in Nodesdf.iterrows():
        # G.add_node(row[0],pos=(row[1],row[2]),ntype=row[3])
        G.add_node(row[0],x=row[1],y=row[2],ntype=row[3])

    #add edges to graph
    for i,row in Edgesdf.iterrows():
        G.add_edge(row[1],row[2],length=row[3])

    # numberOfNodes = G.number_of_nodes()
    # numberOfEdges = G.number_of_edges()
    # numberOfShelters = Sheltersdf.shape[0]
    # print(f'Number of nodes: {numberOfNodes}')
    # print(f'Number of edges: {numberOfEdges}')
    # print(f'Number of shelters: {numberOfShelters}')
    
    #set graph crs
    G.graph['crs'] = {'init': 'epsg:4612'}
    # plot_graph(G)
    # deadends = list(nx.isolates(G))
    # G.remove_nodes_from(deadends)
    # for i, row in Nodesdf.iterrows():
    #     if row[0] in deadends:
    #         Nodesdf.drop(row[0],inplace=True)
    #         Edgesdf = Edgesdf[~Edgesdf['Node1'].isin(deadends)]
    #         Edgesdf = Edgesdf[~Edgesdf['Node2'].isin(deadends)]
    #         continue
    
    # Nodesdf.reset_index(drop=True,inplace=True)
    # Edgesdf.reset_index(drop=True,inplace=True)
    
    Sheltersdf = Nodesdf[Nodesdf["Evacuation"] == 1]
    #calculate shortest paths lengths to each shelter and add to DF
    for i,shrow in Sheltersdf.iterrows():
        lsh = []
        for j,nrow in Nodesdf.iterrows():
            try:
                lsh.append(nx.shortest_path_length(G,source=nrow[0],target=shrow[0],weight='Length'))
            except:# nx.NetworkXNoPath:
                lsh.append(1.0e+10)
        Nodesdf[str(i)]=lsh
        lsh = []
    
    #calculate from each node its closest shelter
    ShelterColumnsSeries = []
    for i, nrow in Nodesdf.iterrows():
        ShelterColumnsSeries.append(nrow[5:].idxmin())
    Nodesdf['shelter']= ShelterColumnsSeries

    #calculate shortest path from each node to corresponding shelter
    ln = []
    for i, nrow in Nodesdf.iterrows():
        try:
            if int(nrow[0]) != int(nrow.shelter):
                ln.append(nx.shortest_path(G,int(nrow[0]),int(nrow.shelter))[1])
            else:
                ln.append(int(nrow[0]))
        except:# nx.NetworkXNoPath:
            ln.append(int(-9999))
    Nodesdf['nextnode']=ln

    data = Nodesdf[['#number','nextnode']]
    data.to_csv(f'./{foldername}/nextnode.csv',index=False)
    
    # Nodesdf.to_csv(f'./{foldername}/nodes.csv',index=False)
    # Edgesdf.to_csv(f'./{foldername}/edges.csv',index=False)

def plot_graph(G):
    """
    Plots a graph using matplotlib from a NetworkX graph object.

    This function converts a MultiGraph to a simple Graph, extracts node coordinates,
    and plots the nodes and edges of the graph.

    Parameters:
    G (networkx.Graph or networkx.MultiGraph): The input graph to be plotted.

    Returns:
    None
    """

    #convert multigraph to graph
    Gnew = nx.Graph(G)
    #to plot an undirected graph
    Gnew_nodes = pd.DataFrame(Gnew.nodes(data=True))
    print(Gnew_nodes.head())
    Gnew_nodes['y']=Gnew_nodes[1].apply(lambda x: x['y']).astype(float)
    Gnew_nodes['x']=Gnew_nodes[1].apply(lambda x: x['x']).astype(float)
    Gnew_nodes['osmid']=Gnew_nodes[0].astype(int)
    Gnew_nodes.drop(1,axis=1,inplace=True)
    Gnew_nodes.index = Gnew_nodes[0]
    # plot a network graph from a pandas dataframe Gnew_nodes with x and y coordinates using matplotlib
    plt.figure(figsize=(10,10))
    plt.scatter(Gnew_nodes['x'],Gnew_nodes['y'],s=3, c='b')
    # add the edges of the graph
    for edge in Gnew.edges():
        x = [Gnew_nodes.loc[edge[0]]['x'],Gnew_nodes.loc[edge[1]]['x']]
        y = [Gnew_nodes.loc[edge[0]]['y'],Gnew_nodes.loc[edge[1]]['y']]
        plt.plot(x,y,'k-',linewidth=0.5)
    plt.show()
    
#erase folders
def erase_folders(case):
    """
    Deletes specified folders related to a given case.

    This function attempts to remove two directories:
    1. A directory named after the provided case.
    2. A directory named after the provided case with the suffix '_CAREFUL_PREVIOUS_INPUT'.

    If the directories do not exist or cannot be removed, the function will silently fail without raising an exception.

    Parameters:
    case (str): The name of the case used to identify the directories to be removed.

    Returns:
    None
    """

    try:
        os.system(f'rm -r ./{case}')
    except:
        pass
    try:
        os.system(f'rm -r ./{case}_CAREFUL_PREVIOUS_INPUT')
    except:
        pass

def clean_short_links(foldername):
    # Load data from CSV files
    links = np.genfromtxt(f'./{foldername}/edges0.csv', delimiter=",", skip_header=1)
    nodes = np.genfromtxt(f'./{foldername}/nodes0.csv', delimiter=",", skip_header=1)

    # Define the length threshold below which links should be removed and nodes merged
    length_threshold = 5

    # Initialize lists to store updated nodes and links
    updated_nodes = []
    updated_links = []

    # Track node mappings to handle merging
    node_mapping = {}  # Dictionary to track merged nodes

    # Process each link
    for link in links:
        if link[3] < length_threshold:  # Check if 'Length' is below threshold (index 3)
            # Get the two nodes connected by this link
            node1, node2 = int(link[1]), int(link[2])  # Node1 and Node2 (indices 1 and 2)

            # Find coordinates of the nodes
            node1_data = nodes[nodes[:, 0] == node1][0]  # node1 data
            node2_data = nodes[nodes[:, 0] == node2][0]  # node2 data
            
            # Calculate the merged coordinates (average position)
            merged_x = (node1_data[1] + node2_data[1]) / 2  # Coord_x (index 1)
            merged_y = (node1_data[2] + node2_data[2]) / 2  # Coord_y (index 2)

            # Update node1's coordinates to the merged position
            updated_nodes.append([node1, merged_x, merged_y, node1_data[3], node1_data[4]])

            # Register node2 in the node mapping dictionary to point to node1
            node_mapping[node2] = node1  # node2 is merged into node1
        else:
            # Keep links that don’t meet the threshold condition
            updated_links.append([link[0], link[1], link[2], link[3], link[4]])

    # Update links based on the node mappings
    for link in updated_links:
        link[1] = node_mapping.get(link[1], link[1])  # Update Node1 if mapped
        link[2] = node_mapping.get(link[2], link[2])  # Update Node2 if mapped

    # Add remaining nodes that were not merged
    remaining_nodes = [node for node in nodes if node[0] not in node_mapping]
    updated_nodes.extend(remaining_nodes)

    # Convert lists back to numpy arrays
    updated_links = np.array(updated_links)
    updated_nodes = np.array(updated_nodes)

    # Create a new mapping for the updated node IDs, starting from 0
    new_id_mapping = {old_id: new_id for new_id, old_id in enumerate(updated_nodes[:, 0])}

    # Apply the new ID mapping to update the node IDs in both `updated_nodes` and `updated_links`
    updated_nodes[:, 0] = np.arange(len(updated_nodes))
    for link in updated_links:
        link[1] = new_id_mapping.get(link[1], link[1])  # Update Node1 in links
        link[2] = new_id_mapping.get(link[2], link[2])  # Update Node2 in links

    # Update the first column in `updated_links` to be a sequential counter starting from 0
    updated_links[:, 0] = np.arange(len(updated_links))

    # Save the cleaned data back to CSV files
    np.savetxt(f'./{foldername}/edges.csv', updated_links, delimiter=",", header="#number,Node1,Node2,Length,Width", comments='', fmt='%d,%d,%d,%.1f,%d')
    np.savetxt(f'./{foldername}/nodes.csv', updated_nodes, delimiter=",", header="#number,Coord_x,Coord_y,evacuation,reward", comments='', fmt='%d,%.6f,%.6f,%d,%d')

    print("Short links removed, nodes merged, counters updated, and link references adjusted successfully.")
    

def makeAdjacencyMatrix(nodesFile= os.path.join("Input","nodes.csv"), 
                        linksFile= os.path.join("Input","edges.csv") ):
    nodedb= np.loadtxt(nodesFile, delimiter=",", skiprows=1)
    linkdb= np.loadtxt(linksFile, delimiter= ",", skiprows=1)
    numNodes= nodedb.shape[0]
    A= np.zeros((numNodes, numNodes))
    for l in linkdb:
        node1= int( l[1] )
        node2= int( l[2] )
        length= l[3]
        A[node1, node2]= length
        A[node2, node1]= length
    return csr_matrix(A)

def makeClosePath(foldername, nodesFile= os.path.join("Input","nodes.csv"), 
                  linksFile= os.path.join("Input","edges.csv") ):
    A= makeAdjacencyMatrix(nodesFile, linksFile)
    dist_matrix, predecessors = dijkstra(csgraph=A, 
                                         directed=False, 
                                         return_predecessors=True)
    
    nodedb= np.loadtxt(nodesFile, delimiter=",", skiprows=1)
    evac_nodes= np.where( nodedb[:,3] == 1 )[0]
    numNodes= nodedb.shape[0]
    nextNodeDB= np.zeros((numNodes,2))
    
    predecessors[np.arange(numNodes),np.arange(numNodes)]= np.arange(numNodes)
    
    for n in range(numNodes):
        dist= dist_matrix[n,evac_nodes]
        node_target= evac_nodes[ np.argmin(dist) ]
        next_node= predecessors[node_target,n]
        nextNodeDB[n,1]= next_node
    
    nextNodeDB[:,0]= np.arange(numNodes)
    
    np.savetxt(os.path.join(foldername,"nextnode.csv"), 
                nextNodeDB, delimiter=",", fmt= "%d")
    return

def create_html_map(point_layers, polygon_layers, foldername):
        #create an interactive map with the evacuation buildings and shelters and inundation
    def create_map(location=[33.5, 133.5]):
        # Create a map centered around `location`
        m = folium.Map(location=location, zoom_start=12)
        return m

    def add_point_layer(m, gdf, color='blue', icon='info-sign'):
        for _, row in gdf.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)

    def add_polygon_layer(m, gdf, color = 'blue'):
        folium.GeoJson(
            gdf.geometry,
            style_function=lambda x: {'fillColor': color, 'color': color}
        ).add_to(m)

    def add_layer_control(m):
        folium.LayerControl().add_to(m)

    # Save the map to an HTML file
    m = create_map()
    if point_layers:
        for layer in point_layers:
            add_point_layer(m, layer['gdf'], color=layer['color'], icon=layer['icon'])
    if polygon_layers:
        for layer in polygon_layers:
            add_polygon_layer(m, layer['gdf'], color=layer['color'])
    add_layer_control(m)
    m.save(f'./{foldername}/{foldername}_map.html')
    return
    
def main(areas, foldername, evacnodes, times, pop, multi=True, use_seed=True, sp=False):
    erase_folders(foldername)
    #prepare folders
    check_folders(foldername, sp=sp)
    #download network
    G = download_nwk(areas[foldername],show=False, close=True, save=True, foldername=foldername)
    # G = download_point_nwk(areas[foldername][0],areas[foldername][1],1000,show=False, close=True, save=True, foldername=foldername)
    #set seed for replication
    if use_seed:
        np.random.seed(10)
    pd.DataFrame(G.nodes(data=True)).to_csv(f'./{foldername}/original_nodes.csv',index=True)
    #choose randomnly 'size' nodes for evacuation
    if evacnodes == None:
        evacnodes = np.random.choice(G.nodes, size=1)
    #prepare databases
    Gdfn = get_nodedb(G, evacnodes=evacnodes, foldername=foldername)
    Gdfe = get_edgedb(G, Gdfn, foldername=foldername)
    clean_short_links(foldername=foldername)
    nodesfile = f'./{foldername}/nodes.csv'
    linksfile= f'./{foldername}/edges.csv' 
    # get_shortpath_file(foldername=foldername)
    makeClosePath(foldername=foldername, nodesFile=nodesfile, linksFile=linksfile)
    if multi:
        for time in times:
            get_fake_population_times(times=time, pop=pop, foldername=foldername)
    else:
        get_fake_population(pop=pop,foldername=foldername)
    get_actionsdb_transitionsdb(foldername=foldername)
    
    

if __name__ == "__main__":
    pass