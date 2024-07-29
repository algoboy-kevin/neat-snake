from render import GAME_W, NET_W, NODE_SIZE, SCREEN_WIDTH, WINDOW_BUFFER


def get_node_centers(net, genome, hidden_nodes):
  
    node_centers = {}

    startX = WINDOW_BUFFER + 75
    startY = WINDOW_BUFFER + NODE_SIZE

    for i, input_node in enumerate(net.input_nodes):
      center2 = startX + 5.5 * NODE_SIZE, startY + i * 3 * NODE_SIZE + 10
      node_centers[input_node] = center2

    startX = WINDOW_BUFFER + 0.5 * NET_W
    startY = WINDOW_BUFFER + NODE_SIZE * 6

    for i, hidden_node in enumerate(hidden_nodes):
      x = startX + 2 * NODE_SIZE if i % 2 == 0 else startX - 2 * NODE_SIZE
      if i == 2: x += NODE_SIZE * 3
      center2 = x, startY + i * 5 * NODE_SIZE + 10
      node_centers[hidden_node] = center2


    startX = SCREEN_WIDTH - GAME_W - WINDOW_BUFFER * 3 - NODE_SIZE + 425
    startY = WINDOW_BUFFER + 12 * NODE_SIZE

    for i, output_node in enumerate(net.output_nodes):
        center2 = startX - 2 * NODE_SIZE, startY + i * 3 * NODE_SIZE + 10
        node_centers[output_node] = center2

    return node_centers


def modify_eval_functions(net, genome, config):
    """
    Modify neat-python's function to display more hidden nodes 
    """
    # Gather expressed connections.
    connections = [cg.key for cg in genome.connections.values() if cg.enabled]

    layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections, genome)
    node_evals = []
    for layer in layers:
        for node in layer:
            inputs = []
            for conn_key in connections:
                inode, onode = conn_key
                if onode == node:
                    cg = genome.connections[conn_key]
                    inputs.append((inode, cg.weight))

            ng = genome.nodes[node]
            aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
            activation_function = config.genome_config.activation_defs.get(ng.activation)
            node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))
  
    return node_evals

def feed_forward_layers(inputs, outputs, connections, genome):
  """
  Modify neat-python's function to display more hidden nodes 
  """
  required = set(genome.nodes)

  layers = []
  s = set(inputs)
  while 1:
      # Find candidate nodes c for the next layer.  These nodes should connect
      # a node in s to a node not in s.
      c = set(b for (a, b) in connections if a in s and b not in s)
      # Keep only the used nodes whose entire input set is contained in s.
      t = set()
      for n in c:
          if n in required and all(a in s for (a, b) in connections if b == n):
              t.add(n)

      if not t:
          break

      layers.append(t)
      s = s.union(t)

  return layers

