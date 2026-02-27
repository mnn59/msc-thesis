"""
Environment Utilities for GAT-MAPPO-DRL-OR
============================================
Extracts topology information from DRL-OR's NetEnv for GAT processing.

DRL-OR's simenv.py is used AS-IS (no modifications needed).
This module provides helper functions to extract the adjacency matrix
and edge list from the existing NetEnv after setup() is called.
"""

import torch


def extract_adjacency_matrix(envs, num_node):
    """
    Extract adjacency matrix from DRL-OR's NetEnv instance.
    
    NetEnv stores topology in:
        envs._link_lists[i] = list of neighbor node indices for node i
        envs._link_capa[i][j] = capacity of link i->j (0 if no link)
    
    Args:
        envs: NetEnv instance (after setup() has been called)
        num_node: number of nodes in the network
    
    Returns:
        adj_matrix: [num_node, num_node] FloatTensor with self-loops
        edge_list: list of (src, dst) tuples
    """
    adj = torch.zeros(num_node, num_node)
    edge_list = []
    
    # Self-loops (each node attends to itself in GAT)
    for i in range(num_node):
        adj[i, i] = 1.0
    
    # Extract from link_lists (available after setup)
    if hasattr(envs, '_link_lists'):
        for src in range(num_node):
            for dst in envs._link_lists[src]:
                adj[src, dst] = 1.0
                edge_list.append((src, dst))
    elif hasattr(envs, '_link_capa'):
        # Fallback: use capacity matrix
        for i in range(num_node):
            for j in range(num_node):
                if envs._link_capa[i][j] > 0:
                    adj[i, j] = 1.0
                    edge_list.append((i, j))
    else:
        # Last resort: fully connected
        print("WARNING: Could not extract topology, using fully connected graph")
        adj = torch.ones(num_node, num_node)
        for i in range(num_node):
            for j in range(num_node):
                if i != j:
                    edge_list.append((i, j))
    
    return adj, edge_list


def get_agent_node_mapping(envs):
    """
    Extract agent-to-node and node-to-agent mappings from NetEnv.
    
    Returns:
        agent_to_node: list mapping agent_id -> node_id
        node_to_agent: list mapping node_id -> agent_id (or None)
    """
    return envs._agent_to_node, envs._node_to_agent