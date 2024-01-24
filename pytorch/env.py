import numpy as np
import random
from dataclasses import dataclass, field
from utils import *


@dataclass
class Env:
    """knowledge graph environment definition"""
    entity2vec: np.ndarray
    relation2vec: np.ndarray
    kb_all: list
    eliminated_kb: list
    entity2id: dict
    relation2id: dict
    relations: list
    task: str
    path: list = field(default_factory=list)
    path_relations: list = field(default_factory=list)
    die: int = 0

    def __init__(self, entity2vec, relation2vec, kb_all, eliminated_kb, entity2id, relation2id, relations, task=None):
        # 将entity2id.txt和relation2id.txt从Env __init__中读取更改为从外部读取。
        # f1 = open(dataPath + 'entity2id.txt')
        # f2 = open(dataPath + 'relation2id.txt')
        # self.entity2id = f1.readlines()
        # self.relation2id = f2.readlines()
        # f1.close()
        # f2.close()
        # self.entity2id_ = {}
        # self.relation2id_ = {}
        # self.relations = []
        # for n, line in enumerate(self.entity2id):
        # 	self.entity2id_[line.split()[0]] =int(line.split()[1])
        # for line in self.relation2id:
        # 	self.relation2id_[line.split()[0]] = int(line.split()[1])
        # 	self.relations.append(line.split()[0])
        self.entity2id_ = entity2id
        self.relation2id_ = relation2id
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.relations = relations

        # self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')
        self.entity2vec = entity2vec
        # self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')
        self.relation2vec = relation2vec

        self.path = []
        self.path_relations = []

        # Knowledge Graph for path finding
        self.kb_all = kb_all

        # self.kb = []
        # if task != None:
        # 	relation = task.split()[2]
        # 	for line in kb_all:
        # 		rel = line.split()[2]
        # 		if rel != relation and rel != relation + '_inv':
        # 			self.kb.append(line)
        self.eliminated_kb = eliminated_kb

        self.die = 0  # record how many times does the agent choose an invalid path

    def interact(self, state, action):
        """
        This function process the interact from the agent
        state: is [current_position, target_position] 
        action: an integer
        return: (reward, [new_postion, target_position], done)
        """
        done = 0  # 是否结束标志
        curr_pos = state[0]
        target_pos = state[1]
        chosen_relation = self.relations[action]
        choices = []
        for line in self.eliminated_kb:  #扫描所有消除的kb
            triple = line.rsplit()
            e1_idx = self.entity2id[triple[0]]

            # 将正确的选择放入choices中
            if curr_pos == e1_idx and triple[2] == chosen_relation and triple[1] in self.entity2id:
                choices.append(triple)
        if len(choices) == 0:
            reward = -1
            self.die += 1
            next_state = state  # stay in the initial state
            next_state[-1] = self.die
            return reward, next_state, done
        else:  # find a valid step
            path = random.choice(choices)
            # 选择正确的三元组并将其附加到self.path中。
            self.path.append(path[2] + ' -> ' + path[1])

            self.path_relations.append(path[2])
            # print('Find a valid step', path)
            # print('Action index', action)
            self.die = 0
            new_pos = self.entity2id[path[1]]
            reward = 0
            new_state = [new_pos, target_pos, self.die]

            if new_pos == target_pos:
                print(('Find a path:', self.path))
                done = 1
                reward = 0
                new_state = None
            return reward, new_state, done

    def get_valid_actions(self, entityID):
        actions = set()
        for line in self.eliminated_kb:
            triple = line.split()
            e1_idx = self.entity2id[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id[triple[2]])
        return np.array(list(actions))

    def path_embedding(self, path, dim):
        embeddings = [self.relation2vec[self.relation2id[relation], :]
                      for relation in path]
        embeddings = np.reshape(embeddings, (-1, dim))
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, (-1, dim))
