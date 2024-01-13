import torch
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def explain_attention(df, graph, model, topk=20):
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
            #visualize_attention_single(top_triplets_with_scores, attention_top_values)
            #visualize_attention(top_triplets_with_scores, attention_top_values, use_neg_synergy=False)
            visualize_synergy2(top_triplets_with_scores, attention_top_values)
            #visualize_attention(top_triplets_with_scores, attention_top_values)
            #visualize_attention2(attention[:,head], top_indices, top_triplets_with_scores)
        return

def visualize_attention(top_triplets_with_scores, attention_top_values, use_neg_synergy=True):

    # Assuming top_triplets_with_scores contains the top triplets and their synergy scores
    # and attention_top_values contains the top attention values
    # Create a graph
    G = nx.Graph()

    # Add nodes and edges with attributes
    for i, ((drug_1, drug_2, context), synergy_score) in enumerate(top_triplets_with_scores):
        G.add_node(f'{drug_1}', type='drug')
        if drug_1==drug_2:
            label = 'Self'
        else:
            label = f'{context}'
            G.add_node(f'{drug_2}', type='drug')
        G.add_edge(f'{drug_1}', f'{drug_2}', weight=attention_top_values[i], synergy=synergy_score, context=label)
    
    if use_neg_synergy:
        # Normalize synergy scores and set up edge attributes for width and style
        baseline_width = 1  # Minimum width
        width_factor = 4    # Factor to increase/decrease the width based on synergy score
        min_synergy, max_synergy = min(score for _, score in top_triplets_with_scores), max(score for _, score in top_triplets_with_scores)

        for u, v, data in G.edges(data=True):
            synergy_score = data['synergy']
            # Normalize the synergy score and apply to width
            if synergy_score >= 0:
                data['width'] = baseline_width + width_factor * (synergy_score / max_synergy)
                data['style'] = 'solid'
            else:
                data['width'] = baseline_width + width_factor * (synergy_score / min_synergy)
                data['style'] = 'dotted'

        # Prepare for drawing
        edge_colors = [plt.cm.viridis(G[u][v]['weight']) for u, v in G.edges()]
        edge_widths = [G[u][v]['width'] for u, v in G.edges()]
        edge_styles = [G[u][v]['style'] for u, v in G.edges()]
    else:
        # Define color map and edge width based on attention weights and synergy scores
        edge_colors = [plt.cm.viridis(G[u][v]['weight'] / max(attention_top_values)) for u, v in G.edges()]
        edge_widths = [2 + 5 * (G[u][v]['synergy'] / max([e[2]['synergy'] for e in G.edges(data=True) if e[2]['synergy'] is not None])) for u, v in G.edges()]

    # Draw the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(G)  # Layout for the nodes

    # Drawing nodes and edges separately to apply color and width mapping to edges
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    if use_neg_synergy:
        # Draw edges with style based on synergy score
        for (u, v), color, width, style in zip(G.edges(), edge_colors, edge_widths, edge_styles):
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], width=width, edge_color=color, style=style)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors)

    # Draw edge labels (context)
    edge_labels = nx.get_edge_attributes(G, 'context')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    # Create a colorbar
    norm = mcolors.Normalize(vmin=0, vmax=max(attention_top_values))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # This line is necessary for ScalarMappable
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Attention Weight')

    plt.title('Visualization of Drug Interactions with Attention Weights and Synergy Scores')
    plt.axis('off')
    plt.show()

def visualize_synergy(top_triplets_with_scores, attention_top_values):

    # Assuming top_triplets_with_scores contains the top triplets and their synergy scores
    # and attention_top_values contains the top attention values
    # Create a graph
    G = nx.Graph()

    # Add nodes and edges with attributes
    for i, ((drug_1, drug_2, context), synergy_score) in enumerate(top_triplets_with_scores):
        G.add_node(f'Drug {drug_1}', type='drug')
        if drug_1==drug_2:
            label = 'Self'
        else:
            label = f'CCL {context}'
        G.add_edge(f'Drug {drug_1}', f'Drug {drug_2}', weight=attention_top_values[i], synergy=synergy_score, context=label)
    
    # Normalize attention weights for edge width
    width_factor = 5  # Adjust the scaling factor as needed
    max_attention = max(attention_top_values)
    edge_widths = [width_factor * weight / max_attention for weight in attention_top_values]

    # Prepare color maps for synergy scores
    positive_cmap = plt.cm.Blues
    negative_cmap = plt.cm.Reds

    # Apply color maps based on synergy scores
    edge_colors = []
    min_synergy, max_synergy = min(score for _, score in top_triplets_with_scores), max(score for _, score in top_triplets_with_scores)
    min_synergy = min(min_synergy,-0.1)
    max_synergy = max(max_synergy,0.1)
    for _, synergy_score in top_triplets_with_scores:
        if synergy_score >= 0:
            # Scale the color intensity within the range of positive synergy scores
            color = positive_cmap(synergy_score / max(max_synergy, 0.1))
        else:
            # Scale the color intensity within the range of negative synergy scores
            color = negative_cmap(1 - abs(synergy_score) / abs(min_synergy))
        edge_colors.append(color)
    
    # Draw the graph
    # Create figure and axes for graph and colorbars
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.5, 8, 0.5])
    ax_graph = fig.add_subplot(gs[0, 1])
    ax_cbar_left = fig.add_subplot(gs[0, 0])
    ax_cbar_right = fig.add_subplot(gs[0, 2])
    pos = nx.spring_layout(G)  # Layout for the nodes

    # Draw nodes and labels
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax_graph)

    # Draw edges
    for (u, v), color, width in zip(G.edges(), edge_colors, edge_widths):
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=[(u, v)], width=width, edge_color=color)

    # Draw context labels
    edge_labels = nx.get_edge_attributes(G, 'context')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax_graph, font_size=8)

    # Colorbars for synergy scores
    # Positive synergy colorbar (blue)
    positive_sm = plt.cm.ScalarMappable(cmap=positive_cmap, norm=plt.Normalize(vmin=0, vmax=max_synergy))
    positive_sm.set_array([])
    cbar_pos = fig.colorbar(positive_sm, cax=ax_cbar_right, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_pos.set_label('Positive Synergy Scores')

    # Negative synergy colorbar (red)
    negative_sm = plt.cm.ScalarMappable(cmap=negative_cmap, norm=plt.Normalize(vmin=min_synergy, vmax=0))
    negative_sm.set_array([])
    cbar_neg = fig.colorbar(negative_sm, cax=ax_cbar_left, orientation='vertical', fraction=0.046, pad=0.08)
    cbar_neg.set_label('Negative Synergy Scores')


    plt.title('Visualization of Drug Interactions with Attention Weights and Synergy Scores')
    plt.axis('off')
    plt.show()



def visualize_synergy2(top_triplets_with_scores, attention_top_values):

    # Assuming top_triplets_with_scores contains the top triplets and their synergy scores
    # and attention_top_values contains the top attention values
    # Create a graph
    G = nx.DiGraph()

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
    # Draw each edge with a unique curvature
    for i, ((drug_1, drug_2, context), _) in enumerate(top_triplets_with_scores):
        if drug_1==drug_2:
            label='Self'
        else:
            label=f'{context}'
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

    #for (u, v), color, width in zip(G.edges(), edge_colors, edge_widths):
    #    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], width=width, edge_color=color)

    # Draw context labels
    edge_labels = nx.get_edge_attributes(G, 'context')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    # Create a colorbar
    norm = mcolors.Normalize(vmin=min_synergy, vmax=max_synergy)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This line is necessary for ScalarMappable
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Synergy score')

    plt.title('Visualization of Drug Interactions with Attention Weights and Synergy Scores')
    plt.axis('off')
    plt.show()

    ok=2


def visualize_attention_single(top_triplets_with_scores, attention_top_values):

    # Assuming top_triplets_with_scores contains the top triplets and their synergy scores
    # and attention_top_values contains the top attention values
    # Create a graph
    G = nx.Graph()

    # Add nodes and edges with attributes
    for i, ((drug_1, drug_2, context), synergy_score) in enumerate(top_triplets_with_scores):
        G.add_node(f'Drug {drug_1}', type='drug')
        if drug_1==drug_2:
            label = 'Self'
        else:
            label = f'CCL {context}'
            G.add_node(f'Drug {drug_2}', type='drug')
        G.add_edge(f'Drug {drug_1}', f'Drug {drug_2}', weight=attention_top_values[i], synergy=synergy_score, context=label)
    
    
    # Define color map and edge width based on attention weights and synergy scores
    edge_colors = [plt.cm.viridis(G[u][v]['weight'] / max(attention_top_values)) for u, v in G.edges()]
    edge_widths = [2 + 5 * (G[u][v]['synergy'] / max([e[2]['synergy'] for e in G.edges(data=True) if e[2]['synergy'] is not None])) for u, v in G.edges()]

    # Draw the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    pos = nx.spring_layout(G)  # Layout for the nodes

    # Drawing nodes and edges separately to apply color and width mapping to edges
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths)
    
    # Draw edge labels (context)
    edge_labels = nx.get_edge_attributes(G, 'context')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    # Create a colorbar
    norm = mcolors.Normalize(vmin=0, vmax=max(attention_top_values))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # This line is necessary for ScalarMappable
    fig.colorbar(sm, ax=ax, orientation='vertical', label='Attention Weight')

    plt.title('Visualization of Drug Interactions with Attention Weights and Synergy Scores')
    plt.axis('off')
    plt.show()
    ok=2


def visualize_attention2(attention, top_indices, top_triplets_with_scores):

    # Assuming top_triplets_with_scores contains the top triplets and their synergy scores
    # and attention_top_values contains the top attention values
    attention_top_values = [attention[i].item() for i in top_indices]  # Extract top attention values

    # Prepare data for plotting
    triplets_labels = [f'Drug {t[0][0]} & Drug {t[0][1]}, Context {t[0][2]}' for t in top_triplets_with_scores]
    synergy_scores = [t[1] for t in top_triplets_with_scores]

    # Create a color palette for synergy scores
    palette = sns.color_palette("coolwarm", as_cmap=True)
    colors = palette(synergy_scores / max(synergy_scores))  # Normalize the synergy scores for color mapping

    # Plotting
    plt.figure(figsize=(12, 8))
    bars = plt.bar(triplets_labels, attention_top_values, color=colors)
    plt.xlabel('Triplets')
    plt.ylabel('Attention Weight')
    plt.title('Top 10 Attention Weights with Synergy Scores')
    plt.xticks(rotation=45, ha='right')

    # Adding a color bar to indicate synergy scores
    sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(vmin=min(synergy_scores), vmax=max(synergy_scores)))
    plt.colorbar(sm, label='Synergy Score')

    plt.tight_layout()
    plt.show()
