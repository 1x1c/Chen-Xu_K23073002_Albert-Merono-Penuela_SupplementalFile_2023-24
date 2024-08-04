import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering

def construct_graph(data_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for editor in data_df['editor'].unique():
        G.add_node(editor)
    for item_id, grp in data_df.groupby(by="item"):
        editors_of_item = grp.editor.values
        for i, editor in enumerate(editors_of_item):
            for coeditor in editors_of_item[i + 1:]:  
                G.add_edge(editor, coeditor)
    return G

def graph_diameter(G: nx.Graph) -> int:
    return nx.diameter(G)

def average_path_length(G: nx.Graph) -> float:
    return nx.average_shortest_path_length(G)

def clustering_coefficient(G: nx.Graph) -> float:
    return nx.average_clustering(G)

def degree_centrality(G: nx.Graph) -> dict:
    return nx.degree_centrality(G)

def betweenness_centrality(G: nx.Graph) -> dict:
    return nx.betweenness_centrality(G)

def closeness_centrality(G: nx.Graph) -> dict:
    return nx.closeness_centrality(G)

def eigenvector_centrality(G: nx.Graph) -> dict:
    return nx.eigenvector_centrality(G)

def modularity_optimization(G: nx.Graph) -> dict:
    import community as community_louvain
    partition = community_louvain.best_partition(G)
    return partition

def spectral_clustering(G: nx.Graph, k: int = 2) -> dict:
    adjacency_matrix = nx.adjacency_matrix(G)
    adjacency_matrix = adjacency_matrix.astype(np.float32)
    
    sc = SpectralClustering(k, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(adjacency_matrix)
    
    communities = {}
    for node, label in zip(G.nodes(), labels):
        communities[node] = label
    return communities

if __name__ == "__main__":
    import os
    DATA_DIR = "data"
    OUTPUT_DIR = "output"

    num_sample_item = None
    # num_sample_item = 10
    if num_sample_item is None:
        G = construct_graph(pd.read_csv(os.path.join(DATA_DIR, "multi-media_editors", f"2type_editors_full.csv")))
    else:
        G = construct_graph(pd.read_csv(os.path.join(DATA_DIR, "multi-media_editors", f"2type_editors_sample_{num_sample_item}.csv")))


    number_of_nodes = G.number_of_nodes()
    number_of_edges = G.number_of_edges()
    average_degree = sum(dict(G.degree()).values()) / number_of_nodes
    diameter = graph_diameter(G)
    avg_path_length = average_path_length(G)
    clustering_coeff = clustering_coefficient(G)

    degree_c = degree_centrality(G)
    betweenness_c = betweenness_centrality(G)
    closeness_c = closeness_centrality(G)
    eigenvector_c = eigenvector_centrality(G)


    modularity_communities = modularity_optimization(G)

    results = {
        "number_of_nodes": number_of_nodes,
        "number_of_edges": number_of_edges,
        "average_degree": average_degree,
        "diameter": diameter,
        "avg_path_length": avg_path_length,
        "clustering_coeff": clustering_coeff,
        "degree_centrality": degree_c,
        "betweenness_centrality": betweenness_c,
        "closeness_centrality": closeness_c,
        "eigenvector_centrality": eigenvector_c,
        "modularity_communities": modularity_communities,
        # "spectral_communities": spectral_clustering(G, k=3) 
    }

    import pickle
    if num_sample_item is None:
        output_file_path = os.path.join(OUTPUT_DIR, f"graph_stat.pkl")
    else:
        output_file_path = os.path.join(OUTPUT_DIR, f"graph_stat_{num_sample_item}.pkl")
    with open(output_file_path, "wb") as f:
        pickle.dump(results, f)