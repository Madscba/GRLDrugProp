import torch
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def explain_attention(df, graph, attention, n_heads=1, topk=20):
    n_nodes = graph.node_feature.shape[0]
    self_loop_edges = torch.stack([torch.arange(n_nodes, device=graph.device)], dim=1).view(-1, 1).repeat(1, 2)
    self_loop_edge_list = torch.cat([self_loop_edges, torch.ones(n_nodes,1,dtype=torch.int,device=graph.device)*graph.num_relation], dim=1)
    edge_list = torch.cat([graph.edge_list, self_loop_edge_list], dim=0)
    
    node_in, node_out, relation = edge_list.t()

    for head in range(n_heads):
        # Get top 20 attention weights and their indices
        top_indices = torch.topk(attention[:,head], topk).indices[1:]
        attention_top_values = attention[top_indices,head].detach().numpy()

        # Retrieve corresponding triplets
        top_triplets = [(node_in[i].item(), node_out[i].item(), relation[i].item()) for i in top_indices]
        
        # Match with synergy scores
        top_triplets_with_scores = []
        for triplet in top_triplets:
            drug_1_id, drug_2_id, context_id = triplet
            if drug_1_id == drug_2_id:
                synergy_score = 0.1
            else:
                synergy_score = df.loc[
                    (df['drug_1_id'] == drug_1_id) & 
                    (df['drug_2_id'] == drug_2_id) & 
                    (df['context_id'] == context_id), 
                    'synergy_zip_mean'
                ].values[0]
            top_triplets_with_scores.append((triplet, synergy_score))

        
        visualize_synergy(top_triplets_with_scores, attention_top_values)
    return


def explain_ccl_attention(df, graph, attention, n_heads=1, topk=20):
    n_nodes = graph.node_feature.shape[0]
    self_loop_edges = torch.stack([torch.arange(n_nodes, device=graph.device)], dim=1).view(-1, 1).repeat(1, 2)
    self_loop_edge_list = torch.cat([self_loop_edges, torch.ones(n_nodes,1,dtype=torch.int,device=graph.device)*graph.num_relation], dim=1)
    edge_list = torch.cat([graph.edge_list, self_loop_edge_list], dim=0)
    
    node_in, node_out, relation = edge_list.t()

    max_synergy = max(df.loc[:,'synergy_zip_mean'])
    min_synergy = min(df.loc[:,'synergy_zip_mean'])
    synergy_values = (max_synergy, -max_synergy)

    for head in range(n_heads):
        # Get top attention weights and their indices
        top_indices = torch.sort(attention[:,head],descending=True).indices[1:]
        

        for c in range(graph.num_relation):
             # Initialize lists to store filtered triplets and corresponding indices
            filtered_triplets = []
            filtered_indices = []

            # Iterate over the top indices to filter based on relation == c until topk elements are added
            index = 0
            while len(filtered_triplets)<topk:
                i = top_indices[index]
                if relation[i].item() == c:
                    # Add the triplet and index to the respective lists
                    filtered_triplets.append((node_in[i].item(), node_out[i].item(), relation[i].item()))
                    filtered_indices.append(i)
                index += 1

            # Match with synergy scores
            top_triplets_with_scores = []
            for triplet in filtered_triplets:
                drug_1_id, drug_2_id, context_id = triplet
                if drug_1_id == drug_2_id:
                    synergy_score = 0.1
                else:
                    synergy_score = df.loc[
                        (df['drug_1_id'] == drug_1_id) & 
                        (df['drug_2_id'] == drug_2_id) & 
                        (df['context_id'] == context_id), 
                        'synergy_zip_mean'
                    ].values[0]
                top_triplets_with_scores.append((triplet, synergy_score))

            attention_top_values = attention[filtered_indices, head].detach().numpy()
            visualize_synergy(top_triplets_with_scores, attention_top_values, synergy_values)
    return

def visualize_synergy(top_triplets_with_scores, attention_top_values, synergy_values, grouping='ccl'):

    # Create a graph
    G = nx.MultiDiGraph()

    # Add nodes and edges with attributes
    for i, ((drug_1, drug_2, context), synergy_score) in enumerate(top_triplets_with_scores):
        G.add_node(f'{drug_1}', type='drug')
        if drug_1==drug_2:
            label = 'Self'
        else:
            label = f'{context}'
            G.add_node(f'{drug_2}', type='drug')    
        G.add_edge(f'{drug_1}', f'{drug_2}', weight=attention_top_values[i], synergy=synergy_score, context=label)
    
    # Normalize attention weights for edge width
    width_factor = 5  # Adjust the scaling factor as needed
    max_attention = max(attention_top_values)
    edge_widths = [width_factor * weight / max_attention for weight in attention_top_values]

    # Prepare color map for synergy scores
    cmap = mcolors.LinearSegmentedColormap.from_list('Custom', [(0, 'darkred'), (0.25, 'red'), (0.5, 'white'), (0.75, 'blue'), (1, 'darkblue')])

    # Normalize the synergy scores to the [0, 1] range for the colormap
    max_synergy, min_synergy = synergy_values
    #min_synergy, max_synergy = min(score for _, score in top_triplets_with_scores), max(score for _, score in top_triplets_with_scores)
    #min_synergy = min(min_synergy,-0.1)
    #max_synergy = max(max_synergy,0.1)
    normalized_synergy_scores = [(score - min_synergy) / (max_synergy - min_synergy) for _, score in top_triplets_with_scores]
    
    # Apply color maps based on synergy scores
    edge_colors = [cmap(score) for score in normalized_synergy_scores]

    # Draw the graph
    # Create figure and axes for graph and colorbars
    fig, ax = plt.subplots(figsize=(14, 12))
    pos = nx.spring_layout(G)  # Layout for the nodes

    # Draw nodes and labels
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax)

    # Draw edges
    # Create a dictionary to track the number of edges between each pair of nodes
    edge_count = {}
    # Draw each edge with a unique curvature
    for i, ((drug_1, drug_2, context), _) in enumerate(top_triplets_with_scores):
        if drug_1==drug_2:
            label={(str(drug_1),str(drug_2)): 'Self'}
        else:
            label={(str(drug_1),str(drug_2)): f'{context}'}
        
        # Case for multiple edges can exist between nodes
        if grouping=='None':
            # Check if this pair of nodes already has an edge
            if (drug_1, drug_2) not in edge_count and (drug_2, drug_1) not in edge_count:
                edge_count[(drug_1, drug_2)] = 0
            else:
                # If the edge exists in either direction, increment the count
                if (drug_2, drug_1) in edge_count:
                    edge_count[(drug_2, drug_1)] += 1
                else:
                    edge_count[(drug_1, drug_2)] += 1

            # Set the curvature based on the count of edges
            curvature = 0.1 * edge_count.get((drug_1, drug_2), 0)

            # Draw the curved edge
            style = f'arc3, rad={curvature}'
        else:
            style = 'arc3'
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(str(drug_1), str(drug_2))],
                                width=edge_widths[i], edge_color=edge_colors[i], connectionstyle=style)
        if grouping=='None':
            nx.draw_networkx_edge_labels(G, pos, edge_labels=label, ax=ax, font_size=8)

    # Create a colorbar
    norm = mcolors.Normalize(vmin=min_synergy, vmax=max_synergy)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This line is necessary for ScalarMappable
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Synergy score')

    if grouping == 'ccl':
        plt.title(f'Visualization of Drug Interactions with Attention Weights and Synergy Scores for cell line {context}')
    elif grouping == 'driug':
        plt.title(f'Visualization of Drug Interactions with Attention Weights and Synergy Scores for drug {drug_1}')
    else:
        plt.title('Visualization of Drug Interactions with Attention Weights and Synergy Scores')
    plt.axis('off')
    plt.show()