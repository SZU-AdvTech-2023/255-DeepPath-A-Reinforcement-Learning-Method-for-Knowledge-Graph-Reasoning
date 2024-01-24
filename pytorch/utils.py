import time
import torch

import random
from collections import namedtuple, Counter
from functools import lru_cache
import numpy as np
from typing import List, Dict, Tuple
import os
from BFS.KB import KB
from BFS.BFS import bfs

# 超参数
state_dim = 200
action_space = 400
embedding_dim = 100
gamma = 0.99
max_steps = 50
max_steps_test = 50
learning_rate = 0.001  # 原代码 0.001
weight_decay = 0.01

cuda = "cuda:0"
seed_number = 12345
# 增加PPO的超参数(原本的论文中0.2为最优)
clip_ratio = 0.2

# 自定义Adam优化器的 epsilon 默认1e-8，PPO及MMDPG中的默认值为1e-5
custom_eps = 1e-8

# trick : state_normalization
use_state_norm = False

# 用于传入wandb作为参数记录
hyperparameters = {
    "state_dim": state_dim,
    "action_space": action_space,
    "embedding_dim": embedding_dim,
    "gamma": gamma,
    "max_steps": max_steps,
    "max_steps_test": max_steps_test,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "cuda": cuda,
    "seed_number": seed_number,
    "clip_ratio": clip_ratio,
    "custom_eps": custom_eps,
    "use_state_norm": use_state_norm
}


dataset = 'NELL-995'
dataPath = os.path.join(os.path.dirname(__file__), "..", dataset)

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# 增加actor-critic需要的定义
ACTransition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'value'))


# 一共12种任务
relations = [
    "concept_agentbelongstoorganization",
    "concept_athletehomestadium",
    "concept_athleteplaysforteam",
    "concept_athleteplaysinleague",
    "concept_athleteplayssport",
    "concept_organizationheadquarteredincity",
    "concept_organizationhiredperson",
    "concept_personborninlocation",
    "concept_personleadsorganization",
    "concept_teamplaysinleague",
    "concept_teamplayssport",
    "concept_worksfor"
]
# relations = [
#     "concept_agentbelongstoorganization",
#     "concept_athletehomestadium",
#     "concept_athleteplaysforteam",
#     "concept_athleteplaysinleague",
#     "concept_athleteplayssport",
#     "concept_organizationheadquarteredincity",
#     "concept_organizationhiredperson",
#     "concept_personborninlocation",
#     "concept_personleadsorganization",
#     "concept_teamplaysinleague",
#     "concept_teamplayssport",
#     "concept_worksfor"
# ]

modes = ["RL", "PPO", "AC"]


def set_seed(seed=0):
    # 设置随机数种子
    random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    np.random.seed(seed)


set_seed(seed_number)

# 修改：使用os.path.join()函数连接路径，避免不同操作系统的路径分隔符不同的问题
# 新增：PPO模型、AC的路径
def get_path_stats_path(mode: str, relation: str):
    paths_dict = {
        "RL": os.path.join(dataPath, "tasks", relation, "path_stats.txt"),
        "PPO": os.path.join(dataPath, "tasks", relation, "path_stats_PPO.txt"),
        "AC": os.path.join(dataPath, "tasks", relation, "path_stats_AC.txt")
    }
    return paths_dict[mode]


def get_path_to_use_path(mode: str, relation: str):
    paths_dict = {
        "RL": os.path.join(dataPath, "tasks", relation, "path_to_use.txt"),
        "PPO": os.path.join(dataPath, "tasks", relation, "path_to_use_PPO.txt"),
        "AC": os.path.join(dataPath, "tasks", relation, "path_to_use_AC.txt")
    }
    return paths_dict[mode]


def distance(e1, e2):
    return np.sqrt(np.sum(np.square(e1 - e2)))


def compare(v1, v2):
    return sum(v1 == v2)


# @lru_cache(maxsize=512)
def teacher(e1, e2, num_paths, entity2vec, entity2id: dict, relation2id: dict, kb: KB, log=False, state_norm=None):
    try:
        intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)
    except Exception as e:
        print("Cannot find a intermediates", e)
        raise e

    res_entity_lists = []
    res_path_lists = []
    for i in range(num_paths):
        # 修改：不能只是因为单个中间节点就抛错不找剩下中间节点的路径了。
        try:
            suc1, entity_list1, path_list1 = bfs(kb, e1, intermediates[i])
            suc2, entity_list2, path_list2 = bfs(kb, intermediates[i], e2)
            if suc1 and suc2:
                res_entity_lists.append(entity_list1 + entity_list2[1:])
                res_path_lists.append(path_list1 + path_list2)
                # print("entity_list1:", entity_list1)
        except Exception as e:
            if log:
                print(e, "Cannot find a path by intermediates",
                      intermediates[i])

    # input("----")
    # print("res_entity_lists:", res_entity_lists)
    # print()
    # print("res_path_lists:", res_path_lists)
    # print("len(res_entity_lists):", len(res_entity_lists))
    # input("-----")
    # ---------- 清理路径 --------
    res_entity_lists_new = []
    res_path_lists_new = []
    for entities, relations in zip(res_entity_lists, res_path_lists):
        rel_ents = []
        for i in range(len(entities) + len(relations)):
            if i % 2 == 0:
                rel_ents.append(entities[int(i / 2)])
            else:
                rel_ents.append(relations[int(i / 2)])

        entity_stats = list(Counter(entities).items())
        duplicate_ents = [item for item in entity_stats if item[1] != 1]
        duplicate_ents.sort(key=lambda x: x[1], reverse=True)
        for item in duplicate_ents:
            ent = item[0]
            ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
            if len(ent_idx) != 0:
                min_idx = min(ent_idx)
                max_idx = max(ent_idx)
                if min_idx != max_idx:
                    rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
        entities_new = []
        relations_new = []
        for idx, item in enumerate(rel_ents):
            if idx % 2 == 0:
                entities_new.append(item)
            else:
                relations_new.append(item)
        res_entity_lists_new.append(entities_new)
        res_path_lists_new.append(relations_new)


    # print("len(res_entity_lists):", len(res_entity_lists_new))
    # input("-----")

    good_episodes = []
    targetID = entity2id[e2]
    # print("targetID:", targetID)
    for path in zip(res_entity_lists_new, res_path_lists_new):
        good_episode = []
        for i in range(len(path[0]) - 1):
            currID = entity2id[path[0][i]]
            nextID = entity2id[path[0][i + 1]]
            state_curr = [currID, targetID, 0]
            state_next = [nextID, targetID, 0]
            actionID = relation2id[path[1][i]]

            # trick： 状态正则化
            now_state = idx_state(entity2vec, state_curr)
            # print("before",now_state)
            # input("------")
            if state_norm is not None:
                now_state = state_norm(now_state)
                # print("now_state",now_state)

            next_state = idx_state(entity2vec, state_next)
            if state_norm is not None:

                next_state = state_norm(next_state)

            good_episode.append(Transition(
                state=now_state,
                action=actionID,
                next_state=next_state,
                reward=1))
        good_episodes.append(good_episode)
    
    # print("len(good_episodes):", len(good_episodes))
    # input("-----")
    
    # 包含 Transition 列表的集合。Transition 包括 'state'。'state' 向量是（curr，targ-curr）的拼接。
    return good_episodes


# 删除路径中的相同实体，并用 '->' 连接
def path_clean(path):
    rel_ents = path.split(' -> ')
    relations = []
    entities = []
    for idx, item in enumerate(rel_ents):
        if idx % 2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    entity_stats = list(Counter(entities).items())
    duplicate_ents = [item for item in entity_stats if item[1] != 1]
    duplicate_ents.sort(key=lambda x: x[1], reverse=True)
    for item in duplicate_ents:
        ent = item[0]
        ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx) != 0:
            min_idx = min(ent_idx)
            max_idx = max(ent_idx)
            if min_idx != max_idx:
                rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
    return ' -> '.join(rel_ents)


def prob_norm(probs):
    return probs / sum(probs)


def open_entity2id_and_relation2id() -> Tuple[Dict, Dict, List]:
    entity2id = {}
    relation2id = {}
    relations = []
    entity2id_file = os.path.join(dataPath, "entity2id.txt")
    relation2id_file = os.path.join(dataPath, "relation2id.txt")
    with open(entity2id_file) as entity2id_file:
        lines = entity2id_file.readlines()
        for line in lines:
            entity2id[line.split()[0]] = int(line.split()[1])
    with open(relation2id_file) as relation2id_file:
        lines = relation2id_file.readlines()
        for line in lines:
            relation2id[line.split()[0]] = int(line.split()[1])
            relations.append(line.split()[0])
    return entity2id, relation2id, relations


def relation_to_kb(relation: str) -> KB:
    # graphpath = dataPath + "tasks/" + relation + "/" + "graph.txt"
    graphpath = os.path.join(dataPath, "tasks", relation, "graph.txt")
    with open(graphpath) as f:
        graph = f.readlines()
        kb = KB()
        for line in graph:
            ent1, rel, ent2 = line.rsplit()
            kb.addRelation(ent1, rel, ent2)
    return kb


# idx_list: [currID, targetID, 0] 或 [nextID, targetID, 0]
def idx_state(entity2vec, idx_list):
    """
    功能：此函数根据给定的实体向量和索引列表构建状态向量。
    操作：如果索引列表不为 None，则根据索引列表中的当前和目标实体的索引提取相应的实体向量，
    然后计算两者之间的差值，最后将这两个向量连接成一个新的状态向量。如果索引列表为 None，则返回 None。
    """
    if idx_list is not None:
        curr = entity2vec[idx_list[0], :]
        targ = entity2vec[idx_list[1], :]
        # 将结果向量添加一个维度，使其成为一个矩阵。
        return np.expand_dims(np.concatenate((curr, targ - curr)), axis=0)
    else:
        return None


# 测试时间的辅助函数，用来测试性能提升。


def measure_time(func, *args, **kwargs):
    """
    测量函数执行时间的函数

    参数：
    - func: 要测量时间的函数
    - args: 传递给函数的位置参数
    - kwargs: 传递给函数的关键字参数

    返回：
    - elapsed_time: 函数执行时间（秒）
    - result: 函数的返回值
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"函数 '{func.__name__}' 执行时间：{elapsed_time:.6f} 秒")

    return elapsed_time, result


if __name__ == '__main__':
    print(prob_norm(np.array([1, 1, 1])))
    # path_clean('/common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01d34b -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/0lfyx -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/01y67v -> /common/topic/webpage./common/webpage/category -> /m/08mbj5d -> /common/topic/webpage./common/webpage/category_inv -> /m/028qyn -> /people/person/nationality -> /m/09c7w')
