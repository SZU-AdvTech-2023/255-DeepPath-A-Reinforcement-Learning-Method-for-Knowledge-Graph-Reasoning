import os
from typing import List
from dataclasses import dataclass
import numpy as np
import random
from dataclasses import dataclass
from utils import *


@dataclass
class Env:
    dataPath: str
    task: str = None
    entity2id: dict = None
    relation2id: dict = None
    relations: List[str] = None
    entity2vec: np.ndarray = None
    relation2vec: np.ndarray = None 
    kb_all: List[str] = None # kb_env_rl.txt
    die: int = 0 # 记录连续多少次没有找到合适的路径

    relation: str = None # 任务的关系
    kb: List[str] = None # kb_all - eliminated_kb
    eliminated_kb: List[str] = None # kb_all - kb

    # self.path = []
    # self.path_relations = []
    path: List[str] = None
    path_relations: List[str] = None


    def __post_init__(self):
        # 从文件加载实体和关系的向量表示
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.kb_env_rl_path = os.path.join(dataPath, "kb_env_rl.txt")
        self.entity2id_path = os.path.join(dataPath, 'entity2id.txt')
        self.relation2id_path = os.path.join(dataPath, 'relation2id.txt')
        
        self.entity2id = {}
        self.relation2id = {}
        self.relations = []
        self.kb_list = []
        
        self.read_kb_all()
        self.read_entity_relation_data()
        self.read_embeddings()
        
    def read_kb_all(self):
        with open(self.kb_env_rl_path) as f:
            self.kb_all = f.readlines()

    def reset_relation(self, relation):
        self.relation = relation
        self.relation_path = os.path.join(dataPath, "tasks", relation, "train_pos")
        self.filter_kb(relation)
        
    def get_train_data(self):
        # 打开包含训练数据的文件
        f = open(self.relation_path)
        train_data = f.readlines()
        f.close()
        return train_data

    
    def filter_kb(self, relation):
        self.kb = relation_to_kb(relation)
        # print("kb size", (self.kb))
        self.eliminated_kb = []
        # concept = "concept:" + relation # 原本实现有误
        concept = "concept:" + relation.split("_")[-1]
        # print(concept)
        # input("---")
        for line in self.kb_all:
            rel = line.split()[2]
            if rel != concept and rel != concept + "_inv":
                self.eliminated_kb.append(line)        
        print("eliminated_kb size", len(self.eliminated_kb))
        return self.kb,self.eliminated_kb

    def reset_kb_list_by_task(self,task):
        self.kb_list = []
        if task != None:
            relation = task.split()[2]
            for line in self.kb_all:
                rel = line.split()[2]
                if rel != relation and rel != relation + '_inv':
                    self.kb_list.append(line)
        self.path = []
        self.path_relations = []

    def read_entity_relation_data(self):
        with open(self.entity2id_path) as f1, open(self.relation2id_path) as f2:
            entity2id_list = f1.readlines()
            relation2id_list = f2.readlines()

        for line in entity2id_list:
            self.entity2id[line.split()[0]] = int(line.split()[1])

        for line in relation2id_list:
            self.relation2id[line.split()[0]] = int(line.split()[1])
            self.relations.append(line.split()[0])

    def read_embeddings(self):
        self.entity2vec = np.loadtxt(os.path.join(self.dataPath, 'entity2vec.bern'))
        self.relation2vec = np.loadtxt(os.path.join(self.dataPath, 'relation2vec.bern'))

    def interact(self, state, action):
        '''
		This function process the interact from the agent
		state: is [current_position, target_position] 
		action: an integer
		return: (reward, [new_postion, target_position], done)
		'''
        done = 0 # Whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]   
        chosed_relation = self.relations[action]
        choices = []
        for line in self.kb_list:
            triple = line.rsplit()
            e1_idx = self.entity2id[triple[0]]
			
            if curr_pos == e1_idx and triple[2] == chosed_relation and triple[1] in self.entity2id:
                choices.append(triple)
                
        if len(choices) == 0:
            reward = -1
            self.die += 1
            next_state = state # stay in the initial state
            next_state[-1] = self.die
            return (reward, next_state, done)
        else: # find a valid step
            path = random.choice(choices)
            self.path.append(path[2] + ' -> ' + path[1])
            self.path_relations.append(path[2])
            # print('Find a valid step', path)
            # print('Action index', action)
            self.die = 0
            new_pos = self.entity2id[path[1]]
            reward = 0
            new_state = [new_pos, target_pos, self.die]

            if new_pos == target_pos:
                print('Find a path:', self.path)
                done = 1
                reward = 0
                new_state = None
            return (reward, new_state, done)

    def idx_state(self, idx_list):
        if idx_list != None:
            curr = self.entity2vec[idx_list[0],:]
            targ = self.entity2vec[idx_list[1],:]
            return np.expand_dims(np.concatenate((curr, targ - curr)),axis=0)
        else:
            return None

    def get_valid_actions(self, entityID):
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id[triple[2]])
        return np.array(list(actions))

    def path_embedding(self, path):
        embeddings = [self.relation2vec[self.relation2id[relation],:] for relation in path]
        embeddings = np.reshape(embeddings, (-1,embedding_dim))
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding,(-1, embedding_dim))

