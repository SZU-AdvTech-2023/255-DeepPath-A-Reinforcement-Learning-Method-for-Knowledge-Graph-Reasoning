#!/usr/bin/python
"""
将单评估修改为多评估，测试并输出所有关系预测的结果
"""
import sys
import numpy as np
from BFS.KB import *
from sklearn import linear_model
import torch
import torch.nn as nn
import torch.optim as optim
from utils import relations, dataPath,get_path_stats_path,get_path_to_use_path,set_seed,seed_number
import os

set_seed(seed_number)

class ModelPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(ModelPyTorch, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def evaluate_single(relation, dataset="NELL-995",mode = "PPO"):
    # 根据给定的关系，构建知识图谱，并使用训练数据进行路径特征的训练
    def train(kb, kb_inv, named_paths):
        f = open(dataPath_ + '/train.pairs')
        train_data = f.readlines()
        f.close()
        train_pairs = []
        train_labels = []
        for line in train_data:
            e1 = line.split(',')[0].replace('thing$', '')
            e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
            if (e1 not in kb.entities) or (e2 not in kb.entities):
                continue
            train_pairs.append((e1, e2))
            label = 1 if line[-2] == '+' else 0
            train_labels.append(label)
        training_features = []
        for sample in train_pairs:
            feature = []
            for path in named_paths:
                feature.append(
                    int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
            training_features.append(feature)

        input_dim = len(named_paths)
        model = ModelPyTorch(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.RMSprop(model.parameters())

        for epoch in range(300):
            inputs = torch.tensor(training_features, dtype=torch.float32)
            labels = torch.tensor(
                train_labels, dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return model

    # 从文件中获取路径特征

    def get_features(featurePath, feature_stats, relationId_path):
        stats = {}
        f = open(feature_stats)
        path_freq = f.readlines()
        f.close()
        for line in path_freq:
            path = line.split('\t')[0]
            num = int(line.split('\t')[1])
            stats[path] = num
        max_freq = np.max(list(stats.values()))

        relation2id = {}
        f = open(relationId_path)
        content = f.readlines()
        f.close()
        for line in content:
            relation2id[line.split()[0]] = int(line.split()[1])

        useful_paths = []
        named_paths = []
        f = open(featurePath)
        paths = f.readlines()
        f.close()

        for line in paths:
            path = line.rstrip()

            length = len(path.split(' -> '))

            if length <= 10:
                pathIndex = []
                pathName = []
                relations = path.split(' -> ')

                for rel in relations:
                    pathName.append(rel)
                    rel_id = relation2id[rel]
                    pathIndex.append(rel_id)
                useful_paths.append(pathIndex)
                named_paths.append(pathName)

        # print 'How many paths used: ', len(useful_paths)
        return useful_paths, named_paths

    # 评估逻辑，包括模型训练、知识图谱构建、以及测试数据的预测与评估

    def evaluate_logic(dataPath_, featurePath, feature_stats, relationId_path):
        kb = KB()
        kb_inv = KB()

        f = open(dataPath_ + '/graph.txt')
        kb_lines = f.readlines()
        f.close()

        for line in kb_lines:
            e1 = line.split()[0]
            rel = line.split()[1]
            e2 = line.split()[2]
            kb.addRelation(e1, rel, e2)
            kb_inv.addRelation(e2, rel, e1)

        _, named_paths = get_features(
            featurePath, feature_stats, relationId_path)

        model = train(kb, kb_inv, named_paths)

        f = open(dataPath_ + '/sort_test.pairs')
        test_data = f.readlines()
        f.close()
        test_pairs = []
        test_labels = []

        for line in test_data:
            e1 = line.split(',')[0].replace('thing$', '')
            e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
            if (e1 not in kb.entities) or (e2 not in kb.entities):
                continue
            test_pairs.append((e1, e2))
            label = 1 if line[-2] == '+' else 0
            test_labels.append(label)

        aps = []
        query = test_pairs[0][0]
        y_true = []
        y_score = []

        score_all = []

        for idx, sample in enumerate(test_pairs):
            if sample[0] == query:
                features = []
                for path in named_paths:
                    features.append(
                        int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

                inputs = torch.tensor(
                    features, dtype=torch.float32).view(1, -1)
                score = model(inputs).item()

                score_all.append(score)
                y_score.append(score)
                y_true.append(test_labels[idx])
            else:
                query = sample[0]
                count = list(zip(y_score, y_true))
                count.sort(key=lambda x: x[0], reverse=True)
                ranks = []
                correct = 0
                for idx_, item in enumerate(count):
                    if item[1] == 1:
                        correct += 1
                        ranks.append(correct/(1.0+idx_))
                if len(ranks) == 0:
                    aps.append(0)
                else:
                    aps.append(np.mean(ranks))

                y_true = []
                y_score = []
                features = []
                for path in named_paths:
                    features.append(
                        int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))

                inputs = torch.tensor(
                    features, dtype=torch.float32).view(1, -1)
                score = model(inputs).item()

                score_all.append(score)
                y_score.append(score)
                y_true.append(test_labels[idx])

        count = list(zip(y_score, y_true))
        count.sort(key=lambda x: x[0], reverse=True)
        ranks = []
        correct = 0
        for idx_, item in enumerate(count):
            if item[1] == 1:
                correct += 1
                ranks.append(correct/(1.0+idx_))
        aps.append(np.mean(ranks))

        score_label = list(zip(score_all, test_labels))
        score_label_ranked = sorted(
            score_label, key=lambda x: x[0], reverse=True)

        mean_ap = np.mean(aps)
        print(f'{mode} MAP: {mean_ap}')
        return mean_ap

    def bfs_two(e1, e2, path, kb, kb_inv):
        '''the bidirectional search for reasoning'''
        start = 0
        end = len(path)
        left = set()
        right = set()
        left.add(e1)
        right.add(e2)

        left_path = []
        right_path = []
        while (start < end):
            left_step = path[start]
            left_next = set()
            right_step = path[end-1]
            right_next = set()

            if len(left) < len(right):
                left_path.append(left_step)
                start += 1
                # print 'left',start
                # for triple in kb:
                # 	if triple[2] == left_step and triple[0] in left:
                # 		left_next.add(triple[1])
                # left = left_next
                for entity in left:
                    try:
                        for path_ in kb.getPathsFrom(entity):
                            if path_.relation == left_step:
                                left_next.add(path_.connected_entity)
                    except Exception as e:
                        # print 'left', len(left)
                        # print left
                        # print 'not such entity'
                        return False
                left = left_next

            else:
                right_path.append(right_step)
                end -= 1
                for entity in right:
                    try:
                        for path_ in kb_inv.getPathsFrom(entity):
                            if path_.relation == right_step:
                                right_next.add(path_.connected_entity)
                    except Exception as e:
                        # print 'right', len(right)
                        # print 'no such entity'
                        return False
                right = right_next

        if len(right & left) != 0:
            return True
        return False

    # 读取路径特征文件路径和关系到ID的映射文件路径，然后进行评估
    dataPath_ = os.path.join(dataPath, "tasks", relation)
    # featurePath = dataPath_ + '/path_to_use.txt'\
    # feature_stats = dataPath_ + '/path_stats.txt'

    featurePath = get_path_to_use_path(mode,relation)
    feature_stats = get_path_stats_path(mode,relation)

    relationId_path = os.path.join(dataPath, "relation2id.txt")
    # relationId_path = f'../{dataset}/relation2id.txt'

    return evaluate_logic(dataPath_, featurePath, feature_stats, relationId_path)


if __name__ == '__main__':
    dataset = "NELL-995"
    mode = "PPO"
    # dataset = "FB15k-237"
    base_path = os.path.join(os.path.dirname(__file__), "results")
    result_path = os.path.join(
        base_path, f"{dataset}_link_prediction_{mode}_map.csv")
    # 若不存在results文件夹，则创建
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # 若存在结果文件，则删除
    if os.path.exists(result_path):
        os.remove(result_path)

    # 增加表头
    with open(result_path, "a") as f:
        f.write(f"relation,{mode}_map\n")
        f.close()

    aver = 0
    count = 0
    for relation in relations:
        print(relation)

        rl_map = evaluate_single(relation, dataset,mode=mode)
        # 记录relation和rl_map到csv文件中
        # 去除relation的前缀concept_前缀
        relation = relation.split("_")[1]
        with open(result_path, "a") as f:
            f.write(f"{relation},{rl_map}\n")
            f.close()
        aver += rl_map
        count += 1

    print(f"Average {mode} MAP: {aver / count}")
    with open(result_path, "a") as f:
        f.write(f"Average {mode} MAP,{aver / count}\n")
        f.close()

    print("Done!")
