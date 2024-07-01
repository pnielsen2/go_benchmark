class Node():
    def __init__(self,num_possible_moves, current_player,reset_data, parent_node_move_pair=None):
        self.num_possible_moves = num_possible_moves
        # last_hashes=[[0]]*8
        self.next_nodes = {}
        self.outcomes = torch.zeros(self.num_possible_moves,2)
        self.children_nodes_solved = torch.zeros(self.num_possible_moves)
        self.solved = 0
        self.current_player = current_player
        self.network_input = None
        self.predictions = None
        self.previous_posterior_alphas = None
        self.reset_data = reset_data
        self.parents = [] if parent_node_move_pair == None else [parent_node_move_pair]
        self.position_hash = copy.deepcopy(self.reset_data[9])


def choose_move(node, gamesim, run_stats, network):
    if node.network_input==None:
        player_indicator = torch.zeros(1,2,gamesim.dimension,gamesim.dimension)
        player_indicator[0,{'black':0, 'white':1}[node.current_player]] +=1
        network_input = torch.cat((player_indicator, gamesim.input_history[-8:])).view(1,18, gamesim.dimension, gamesim.dimension)
        node.network_input = network_input
    if node.predictions==None:
        node.predictions = (network(node.network_input).squeeze(0) if network != None 
                            else (node.previous_posterior_alphas if node.previous_posterior_alphas != None 
                                  else torch.zeros_like(node.outcomes) + .5))
    game_winner = None
    unsolved_nodes = (node.children_nodes_solved!=-1)
    posterior_alphas = node.outcomes + node.predictions
    indices = pyro.distributions.Beta(posterior_alphas[:,0], posterior_alphas[:,1]).sample()
    indices = indices*unsolved_nodes
    gamesim_set=False
    while True:
        chosen_move = indices.argmax().item()
        move_xy = (chosen_move // gamesim.dimension, chosen_move % gamesim.dimension)
        # if move is not already solved for the opponent:
        if node.children_nodes_solved[chosen_move] == 0:
            if chosen_move not in node.next_nodes:
                if not gamesim_set:
                    gamesim.set(node.reset_data)
                    gamesim_set=True
                if gamesim.step(move_xy):
                    run_stats['transitions_observed'] +=1
                    game_winner = gamesim.winner
                    break
                else:
                    node.children_nodes_solved[chosen_move]=-1
            else:
                break
        indices[chosen_move]=0
        if not indices.any():
            node.solved = -1
            break
    return chosen_move, game_winner

node.position_hash



node_outcomes = buffer.dataset['Test']['y'][buffer.position_hash_map['Test'][node.position_hash]]
node.outcomes + node.network_predictions