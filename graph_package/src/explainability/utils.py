import torch
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def explain_attention(df, graph, model, topk=20, grouping='ccl'):
    n_nodes = graph.node_feature.shape[0]
    self_loop_edges = torch.stack([torch.arange(n_nodes, device=graph.device)], dim=1).view(-1, 1).repeat(1, 2)
    self_loop_edge_list = torch.cat([self_loop_edges, torch.ones(n_nodes,1,dtype=torch.int,device=graph.device)*graph.num_relation], dim=1)
    edge_list = torch.cat([graph.edge_list, self_loop_edge_list], dim=0)
    
    node_in, node_out, relation = edge_list.t()
    for layer in model.gnn_layers:
        _, attention = layer(graph, graph.node_feature, return_att=True)

        for head in range(layer.n_heads):
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

def visualize_synergy(top_triplets_with_scores, attention_top_values):

    # Assuming top_triplets_with_scores contains the top triplets and their synergy scores
    # and attention_top_values contains the top attention values
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
       # G.add_edge(f'{drug_1}', f'{drug_2}', weight=attention_top_values[i], synergy=synergy_score, context=label)
        G.add_edge(f'{drug_1}', f'{drug_2}', key=f'edge_{i}', weight=attention_top_values[i], synergy=synergy_score, context=label)

    
    # Normalize attention weights for edge width
    width_factor = 5  # Adjust the scaling factor as needed
    max_attention = max(attention_top_values)
    edge_widths = [width_factor * weight / max_attention for weight in attention_top_values]

    # Prepare color map for synergy scores
    cmap = mcolors.LinearSegmentedColormap.from_list('Custom', [(0, 'red'), (0.5, 'lightgrey'), (1, 'blue')])

    # Normalize the synergy scores to the [0, 1] range for the colormap
    min_synergy, max_synergy = min(score for _, score in top_triplets_with_scores), max(score for _, score in top_triplets_with_scores)
    min_synergy = min(min_synergy,-0.1)
    max_synergy = max(max_synergy,0.1)
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
    #nx.draw(G, with_labels=True, connectionstyle='arc3, rad = 0.1')

     # Draw edges
    #for i, triplet in enumerate(top_triplets_with_scores): 
    #    u = str(triplet[0][0])
    #    v = str(triplet[0][1])
    #    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], width=edge_widths[i], edge_color=edge_colors[i])

    # Draw edges
    # Create a dictionary to track the number of edges between each pair of nodes
    edge_count = {}
    edge_labels = {}
    # Draw each edge with a unique curvature
    for i, ((drug_1, drug_2, context), _) in enumerate(top_triplets_with_scores):
        if drug_1==drug_2:
            label={(str(drug_1),str(drug_2)): 'Self'}
        else:
            label={(str(drug_1),str(drug_2)): f'{context}'}
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
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(str(drug_1), str(drug_2))],
                               width=edge_widths[i], edge_color=edge_colors[i], connectionstyle=style)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=label, ax=ax, font_size=8)

    #for (u, v), color, width in zip(G.edges(), edge_colors, edge_widths):
    #    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], width=width, edge_color=color)

    # Draw context labels
    #edge_labels = nx.get_edge_attributes(G, 'context')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    # Create a colorbar
    norm = mcolors.Normalize(vmin=min_synergy, vmax=max_synergy)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This line is necessary for ScalarMappable
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Synergy score')

    plt.title('Visualization of Drug Interactions with Attention Weights and Synergy Scores')
    plt.axis('off')
    plt.show()