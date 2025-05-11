import networkx as nx
import pandas as pd
import numpy as np
import math
import json
import datetime
import os
import pickle
from faker import Faker
from openai import OpenAI
from sentence_transformers import util
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables with fallbacks
TRIP_FILE = os.getenv('TRIP_FILE', None)
EMBEDDING_BASE_URL = os.getenv('EMBEDDING_BASE_URL', 'http://localhost:11434/v1')
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY', '123')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'mxbai-embed-large')

class BehaviorGraph:

    def __init__(self,sample_num=1000,trip_file=None,graph_file=None):
        self.num_sample = sample_num
        if not trip_file:
            trip_file = TRIP_FILE
        self.graph_file = graph_file
        if not graph_file:
            self.graph_file = f"cache/graph/graph_{self.num_sample}"
        if self.graph_file and os.path.exists(self.graph_file):
            self.behavior_graph = self.load_graph()
        elif trip_file and os.path.exists(trip_file):
            trip_df = self._load_data(trip_file,sample_num)
            self.behavior_graph = self._create_graph(trip_df)
            self.behavior_graph = self._embedding_graph()
            self.save_graph()
        else:
            print("No graph file or trip file provided.")
        return
    
    def _load_data(self,trip_file,sample_num,random_state=42):

        trip_df = pd.read_csv(trip_file)
        
        def strtime2number(strtime):
            time = datetime.datetime.strptime(strtime, "%H:%M:%S").time()
            return time.hour+time.minute/60

        def durationt2str(duration_time):
            min_duration = int(math.floor(duration_time/10)*10)
            max_duration = int(math.ceil(duration_time/10)*10)
            if min_duration == max_duration:
                max_duration += 10
            return f"{min_duration}-{max_duration}"
        
        def get_fakename(gender):
            fake = Faker()
            if gender=='male':
                return fake.name_male()
            else:
                return fake.name_female()

        trip_df = trip_df.rename(columns={
            'trip_purpose': 'trip_purpose',
            'trip_start_time':'start_time',
            'primary_mode':'primary_mode',
            'trip_duration_minutes':'duration_minutes',
            'destination_land_use':'target_landuse',
            'trip_taker_person_id':'person_id',
            'trip_taker_household_id':'household_id',
            'trip_taker_age': 'age',
            'trip_taker_sex':'gender',
            'previous_trip_purpose':'previous_status',
            'trip_taker_employment_status':'employment_status',
            'trip_taker_household_size':'household_size',
            'trip_taker_household_income':'household_income',
            'trip_taker_available_vehicles':'available_vehicles',
            'trip_taker_industry':'industry',
            'trip_taker_education':'education',
            'trip_taker_work_bgrp_2020':'work_bgrp',
            'trip_taker_home_bgrp_2020':'home_bgrp'
        })

        trip_df['duration_minutes'] = trip_df['duration_minutes'].apply(durationt2str)
        trip_df['start_time'] = trip_df['start_time'].apply(strtime2number)

        trip_df = trip_df[['activity_id','person_id','household_id','age','gender','previous_status','employment_status','household_size','household_income','available_vehicles','industry','education','trip_purpose','start_time','primary_mode','duration_minutes','target_landuse','work_bgrp','home_bgrp']]

        trip_df = trip_df.sample(n=sample_num, random_state=random_state)
        # get fake_name
        trip_df['name'] = trip_df['gender'].apply(lambda x: get_fakename(x))
        # get relatives
        trip_df['relatives'] = trip_df['household_id']
        trip_df['relatives'] = trip_df['relatives'].apply(lambda x: trip_df[trip_df['household_id']==x]['person_id'].to_list())

        return trip_df
    
    def _create_graph(self,trip_df):
        attribute_dict = {
            'desire' : ['trip_purpose','start_time'],
            'intention': ['primary_mode','duration_minutes'],
            'person':['name','age','gender','previous_status','employment_status','household_size','household_income','available_vehicles','education','work_bgrp','home_bgrp']
        }
        behavior_graph = nx.Graph()
        for idx,row in trip_df.iterrows():
            behavior_graph.add_node(
                'Person_'+str(row['person_id']),
                type='person',
                props=row[attribute_dict['person']].to_dict()
            )
            behavior_graph.add_node(
                'Desire_'+str(row['activity_id']),
                type='desire',
                props = row[attribute_dict['desire']].to_dict()
            )
            behavior_graph.add_node(
                'Intention_'+str(row['activity_id']),
                type='intention',
                props=row[attribute_dict['intention']].to_dict()
            )
            behavior_graph.add_edge(
                'Person_'+str(row['person_id']),'Desire_'+str(row['activity_id']),type="want_to"
            )
            behavior_graph.add_edge(
                'Desire_'+str(row['activity_id']),'Intention_'+str(row['activity_id']),type="choose_to"
            )
            for p in row['relatives']:
                if p != row['person_id']:
                    behavior_graph.add_edge('Person_'+str(p),'Person_'+str(row['person_id']),type="relative_of")
        return behavior_graph
    
    def _get_embeddings(self,texts):
            client = OpenAI(
                base_url=EMBEDDING_BASE_URL,
                api_key=EMBEDDING_API_KEY,
            )
            model=EMBEDDING_MODEL
            if not texts:
                return [] 
            else:
                response = client.embeddings.create(input=texts, model=model)
                return [item.embedding for item in response.data]
    
    def _embedding_graph(self, batch_size=20):
        nodes_to_embed = []
        texts_to_embed = []
        for node_id, data in self.behavior_graph.nodes(data=True):
            if data.get('type') == 'person':
                props = data.get('props')
                if props:
                    props_string = json.dumps(props)
                    nodes_to_embed.append(node_id)
                    texts_to_embed.append(props_string)
        all_embedding_vectors = []
        total_texts = len(texts_to_embed)
        num_batches = math.ceil(total_texts / batch_size) # Calculate number of batches needed
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, total_texts)
            batch_texts = texts_to_embed[start_index:end_index]
            batch_nodes = nodes_to_embed[start_index:end_index] # Keep track of corresponding nodes for this batch
            batch_embedding_vectors = self._get_embeddings(batch_texts)

            assert batch_embedding_vectors and len(batch_embedding_vectors) == len(batch_texts)

            all_embedding_vectors.extend(batch_embedding_vectors)

        for i, node_id in enumerate(nodes_to_embed):
            self.behavior_graph.nodes[node_id]['embedding'] = all_embedding_vectors[i]
        return self.behavior_graph
    
    def _similarity_search(self,query_text, k=5):
        # Encode query
        query_embedding = self._get_embeddings(query_text)
        query_embedding = np.array(query_embedding,dtype=np.float32)
        # Collect all person nodes with their text representations
        person_nodes = []
        person_embeddings = []

        for node_id, data in self.behavior_graph.nodes(data=True):
            if data.get('type') == 'person':
                person_nodes.append(node_id)
                person_embeddings.append(data['embedding'])

        person_embeddings = np.array(person_embeddings,dtype=np.float32)
        
        # Perform semantic search using cosine similarity
        hits = util.semantic_search(query_embedding, person_embeddings, top_k=k)[0]
        # Format results
        results = []
        for hit in hits:
            idx = hit['corpus_id']
            score = hit['score']
            results.append((person_nodes[idx], score))

        return results

    def _dfs_search(self,source,depth):
        node_list = []
        layers = nx.bfs_layers(self.behavior_graph,sources=source)
        for idx,layer in enumerate(layers):
            if idx<depth:
                node_list.extend(layer)
        return node_list
    
    def _get_reference_graph(self,query_text,desire='eat',time=8,k=20,depth=4):
        node_list = []
        similar_weight = {}
        person_nodes = self._similarity_search(query_text=query_text, k=k)
        for i, (node_id, score) in enumerate(person_nodes):
            similar_weight[node_id] = score
            related_nodes = self._dfs_search(source=node_id,depth=depth)
            node_list.extend(related_nodes)
        new_graph = self.behavior_graph.subgraph(node_list)

        if desire:
            desire_embedding = self._get_embeddings(desire)
            desire_nodes = [d['props']['trip_purpose'] for n,d in new_graph.nodes(data=True) if d['type']=='desire']
            desire_weights = {}
            for d in desire_nodes:
                d_embed = self._get_embeddings(d)
                d_w = util.cos_sim(d_embed,desire_embedding).item()
                desire_weights[d] = d_w

        # create subgraph
        subgraph = nx.Graph()
        # add agent node
        agent_id = f'Agent_{hash(query_text)}'
        subgraph.add_node(agent_id,type='agent',props=query_text)

        # merge nodes with same intention
        id_mapping = {}
        for node_id,data in new_graph.nodes(data=True):
            node_type = data['type']
            node_props = data['props']
            if node_type == 'desire':
                attr_key = node_id
            else:
                attr_key = hash(json.dumps(node_props))
            id_mapping[node_id]=attr_key
            subgraph.add_node(attr_key,type=node_type,props=node_props)
            if node_type=='person' and node_id in similar_weight.keys():
                subgraph.add_edge(agent_id,attr_key,type="similar_to",weight=similar_weight[node_id])

        # edge weight
        for u,v,data in new_graph.edges(data=True):
            u_attr = id_mapping[u]
            v_attr = id_mapping[v]
            edge_type = data['type']
            if new_graph.nodes[u]['type'] == 'desire':
                    desire_node = u
            elif new_graph.nodes[v]['type'] == 'desire':
                desire_node = v
            if edge_type == 'want_to':
                node_props = new_graph.nodes[desire_node]['props']
                if desire:
                    weight = desire_weights[node_props['trip_purpose']]
                else:
                    weight = 1
            elif edge_type == 'choose_to':
                node_props = new_graph.nodes[desire_node]['props']
                start_time = node_props['start_time']
                weight = abs(time-start_time)/24
            elif edge_type == 'relative_of':
                weight = 1
            
            subgraph.add_edge(u_attr,v_attr,type=edge_type,weight=weight)

        # normalize edge weight
        max_want_weight = max([data['weight'] for u,v,data in subgraph.edges(data=True) if data['type']=='want_to'])
        max_choose_weight = max([data['weight'] for u,v,data in subgraph.edges(data=True) if data['type']=='choose_to'])
        for u,v,data in subgraph.edges(data=True):
            edge_type = data['type']
            edge_weight = data['weight']
            if edge_type == 'want_to':
                subgraph[u][v]['weight'] = edge_weight/max_want_weight
            elif edge_type == 'choose_to':
                subgraph[u][v]['weight'] = edge_weight/max_choose_weight
        return subgraph
    
    def preference_modelling(self,profile,desire='eat',time=12,k=20,depth=4):
        graph = self._get_reference_graph(query_text=profile,desire=desire,time=time,k=k,depth=depth)
        agent_node = [n for n,d, in graph.nodes(data=True) if d['type']=='agent'][0]
        intention_nodes = [n for n,d, in graph.nodes(data=True) if d['type']=='intention']
        weights = []
        for i in intention_nodes:
            all_path = nx.all_simple_paths(graph, source=agent_node, target=i,cutoff=4)
            intention_weight = 0
            for path in all_path:
                path_weight = 1
                for n in range(len(path)-1):
                    path_weight *= graph[path[n]][path[n+1]]['weight']
                intention_weight += path_weight
            node_props = graph.nodes[i]['props']
            node_props['weight'] = intention_weight
            weights.append(node_props)
        # normlization
        sum_weights = sum([w['weight'] for w in weights])
        for w in weights:
            w['weight'] /= sum_weights
        return graph,weights
    
    def save_graph(self):
        if not os.path.exists(os.path.dirname(self.graph_file)):
            os.makedirs(os.path.dirname(self.graph_file))
        with open(self.graph_file,'wb') as f:
            pickle.dump(self.behavior_graph,f)
    
    def load_graph(self):
        with open(self.graph_file,'rb') as f:
            self.behavior_graph = pickle.load(f)
        return self.behavior_graph
    
    def visualize_graph(self, subgraph,desire='eat',time=12,node_size=300,font_size=9,title_size=16):
        # Edge colors by relationship type
        edges = subgraph.edges(data=True)
        edge_colors = []
        for _, _, data in edges:
            edge_type = data.get('type', '')
            if edge_type == 'similar_to':
                edge_colors.append('#FF1744')  # Red
            elif edge_type == 'want_to':
                edge_colors.append('#1f77b4')  # Blue
            elif edge_type == 'choose_to':
                edge_colors.append('#ff7f0e')  # Orange
            elif edge_type == 'relative_of':
                edge_colors.append('#d62728')  # Red
            else:
                edge_colors.append('#7f7f7f')  # Gray

        # Create a figure with two subplots - main graph and explanation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=200, 
                                        gridspec_kw={'width_ratios': [3, 1]})
        plt.sca(ax1)  # Set the main graph as active axis

        # Create node type groups for hierarchical visualization
        hierarchy_pos = {}
        agent_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'agent']
        person_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'person']
        desire_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'desire']
        intention_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'intention']

        # Position nodes in layers
        y_agent,y_person, y_desire, y_intention = 0.8, 0.6, 0.4, 0.2
        x_spacing_agent = 1.0 / (len(agent_nodes) + 1)
        x_spacing_person = 1.0 / (len(person_nodes) + 1)
        x_spacing_desire = 1.0 / (len(desire_nodes) + 1)
        x_spacing_intention = 1.0 / (len(intention_nodes) + 1)

        for i, node in enumerate(agent_nodes):
            hierarchy_pos[node] = (x_spacing_agent * (i + 1), y_agent)

        for i, node in enumerate(person_nodes):
            hierarchy_pos[node] = (x_spacing_person * (i + 1), y_person)
            
        for i, node in enumerate(desire_nodes):
            hierarchy_pos[node] = (x_spacing_desire * (i + 1), y_desire)
            
        for i, node in enumerate(intention_nodes):
            hierarchy_pos[node] = (x_spacing_intention * (i + 1), y_intention)

        # Draw the graph with hierarchical positions
        nx.draw_networkx_nodes(subgraph, hierarchy_pos, 
                            nodelist=agent_nodes, node_color='#FF1744', node_size=node_size, alpha=0.8, ax=ax1)
        nx.draw_networkx_nodes(subgraph, hierarchy_pos, 
                            nodelist=person_nodes, node_color='#1f77b4', node_size=node_size, alpha=0.8, ax=ax1)
        nx.draw_networkx_nodes(subgraph, hierarchy_pos, 
                            nodelist=desire_nodes, node_color='#ff7f0e', node_size=node_size, alpha=0.8, ax=ax1)
        nx.draw_networkx_nodes(subgraph, hierarchy_pos, 
                            nodelist=intention_nodes, node_color='#2ca02c', node_size=node_size, alpha=0.8, ax=ax1)

        # Draw edges
        edge_weights = [d['weight']*2 for n,v, d in subgraph.edges(data=True)]
        
        # Create separate edge collections for the legend
        similar_edge_list = [(u,v) for u,v,d in subgraph.edges(data=True) if d['type']=='similar_to']
        similar_edge_width = [max(0.1,d['weight']*2) for u,v, d in subgraph.edges(data=True) if d['type']=='similar_to']
        similar_edges = nx.draw_networkx_edges(subgraph, hierarchy_pos, 
                                            edgelist=similar_edge_list,
                                            edge_color='#FF1744', width=similar_edge_width, alpha=0.5, label='similar_to', ax=ax1)
        want_edge_list = [(u,v) for u,v,d in subgraph.edges(data=True) if d['type']=='want_to']
        want_edge_width = [max(0.1,d['weight']*2) for u,v, d in subgraph.edges(data=True) if d['type']=='want_to']
        want_edges = nx.draw_networkx_edges(subgraph, hierarchy_pos, 
                                        edgelist=want_edge_list,
                                        edge_color='#1f77b4', width=want_edge_width, alpha=0.5, label='want_to', ax=ax1)
        
        choose_edge_list = [(u,v) for u,v,d in subgraph.edges(data=True) if d['type']=='choose_to']
        choose_edge_width = [max(0.1,d['weight']*2) for u,v, d in subgraph.edges(data=True) if d['type']=='choose_to']
        choose_edges = nx.draw_networkx_edges(subgraph, hierarchy_pos, 
                                            edgelist=choose_edge_list,
                                            edge_color='#ff7f0e', width=choose_edge_width, alpha=0.5, label='choose_to', ax=ax1)
        relative_edge_list = [(u,v) for u,v,d in subgraph.edges(data=True) if d['type']=='relative_of']
        relative_edge_width = [max(0.1,d['weight']*2) for u,v, d in subgraph.edges(data=True) if d['type']=='relative_of']
        relative_edges = nx.draw_networkx_edges(subgraph, hierarchy_pos, 
                                            edgelist=relative_edge_list,
                                            edge_color='#d62728', width=relative_edge_width, alpha=0.5, label='relative_of', ax=ax1)

        # Draw all edges with their weights
        nx.draw_networkx_edges(subgraph, hierarchy_pos, edge_color=edge_colors, width=edge_weights, alpha=0.5, ax=ax1)

        # Draw simplified labels
        simple_labels = {}
        for node in agent_nodes:
            props = subgraph.nodes[node]['props']
            simple_labels[node] = props

        for node in person_nodes:
            name = subgraph.nodes[node]['props']['name'].split(' ')[0]  # Just first name
            simple_labels[node] = name
            
        for node in desire_nodes:
            attrs = subgraph.nodes[node]
            purpose = f"{attrs['props']['trip_purpose']}"
            simple_labels[node] = purpose
            
        for node in intention_nodes:
            attrs = subgraph.nodes[node]
            simple_labels[node] = f"{attrs['props']['primary_mode']}\n{attrs['props']['duration_minutes']} min"

        nx.draw_networkx_labels(subgraph, hierarchy_pos, labels=simple_labels, font_size=font_size, ax=ax1)
        
        # Create legend for nodes and edges
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF1744', markersize=10, label='Agent'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Person'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10, label='Desire'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10, label='Intention'),
            plt.Line2D([0], [0], color='#d62728', lw=2, label='relative_of (weight: relationship closeness)'),
            plt.Line2D([0], [0], color='#FF1744', lw=2, label='similar_to (weight: profile similarity)'),
            plt.Line2D([0], [0], color='#1f77b4', lw=2, label='want_to (weight: desire similarity)'),
            plt.Line2D([0], [0], color='#ff7f0e', lw=2, label='choose_to (weight: time similarity)'),
        ]
        ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0, 1), 
                   ncol=2, frameon=False, labelspacing=1.1)
        # ax1.legend([agent_nodes_plot, person_nodes_plot, desire_nodes_plot, intention_nodes_plot,
        #             similar_edges, want_edges, choose_edges, relative_edges],
        #         ['Agent', 'Person', 'Desire', 'Intention',
        #             'similar_to (weight: profile similarity)', 'want_to (weight: desire similarity)',
        #             'choose_to (weight: time similarity)', 'relative_of'],
        #         loc='upper left', bbox_to_anchor=(0, 1), ncol=2, frameon=False, labelspacing=1.1)
        
        ax1.axis('off')
        
        # Create the explanation diagram in the second subplot
        plt.sca(ax2)
        explanation_graph = nx.DiGraph()
        agent_node = subgraph.nodes[agent_nodes[0]]['props']
        desire_node = desire
        intention_node = "?"
        explanation_graph.add_node(agent_node, type="agent")
        explanation_graph.add_node(desire_node, type="desire")
        explanation_graph.add_node(intention_node, type="intention")
    
        explanation_graph.add_edge(agent_node, desire_node, type="want_to")
        explanation_graph.add_edge(desire_node, intention_node, type="choose_to")
        
        # Position nodes vertically
        explanation_pos = {
            agent_node: (0.5, 0.8),
            desire_node: (0.5, 0.5),
            intention_node: (0.5, 0.2)
        }
        
        # Draw explanation nodes with matching colors to main graph
        nx.draw_networkx_nodes(explanation_graph, explanation_pos, 
                            nodelist=[agent_node], node_color='#FF1744', node_size=node_size*2, alpha=0.9, ax=ax2)
        nx.draw_networkx_nodes(explanation_graph, explanation_pos, 
                            nodelist=[desire_node], node_color='#ff7f0e', node_size=node_size*2, alpha=0.9, ax=ax2)
        nx.draw_networkx_nodes(explanation_graph, explanation_pos, 
                            nodelist=[intention_node], node_color='#2ca02c', node_size=node_size*2, alpha=0.9, ax=ax2)
        
        # Draw explanation edges with consistent colors
        want_to_edge = [(agent_node, desire_node)]
        choose_to_edge = [(desire_node, intention_node)]
        
        nx.draw_networkx_edges(explanation_graph, explanation_pos, 
                            edgelist=want_to_edge, edge_color='#1f77b4', width=2, alpha=0.8, 
                            ax=ax2)
        nx.draw_networkx_edges(explanation_graph, explanation_pos, 
                            edgelist=choose_to_edge, edge_color='#ff7f0e', width=2, alpha=0.8, ax=ax2)
        
        # Add labels to explanation edges
        edge_labels = {
            (agent_node, desire_node): "want_to",
            (desire_node, intention_node): "choose_to"
        }
        nx.draw_networkx_edge_labels(explanation_graph, explanation_pos, edge_labels=edge_labels, ax=ax2, font_size=font_size)
        
        # Add node labels
        nx.draw_networkx_labels(explanation_graph, explanation_pos, font_size=font_size, ax=ax2)
        
        ax2.text(0.2, 0.91, "Agent", 
            ha='center', va='center', fontsize=font_size, transform=ax2.transAxes)
        ax2.text(0.2, 0.5, "Desire", 
            ha='center', va='center', fontsize=font_size, transform=ax2.transAxes)
        ax2.text(0.2, 0.06, "Intention", 
            ha='center', va='center', fontsize=font_size, transform=ax2.transAxes)
        ax2.axis('off')

        fig.suptitle(f"Today is a normal weekday,\nWhat transportation mode would {agent_node} choose when he/she want to {desire_node} at {time}:00?", fontsize=title_size, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return