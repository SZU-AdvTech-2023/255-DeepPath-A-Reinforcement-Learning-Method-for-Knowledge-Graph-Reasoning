#!/usr/bin/python

import os
import numpy as np
import sys
from BFS.KB import *
from utils import get_path_to_use_path, get_path_stats_path, relations


def fact_prediction_single(mode="AC", relation="concept_agentbelongstoorganization",dataset_name="NELL-995"):

    # 文件路径设置
    base_path = os.path.abspath(__file__)
    base_dir = os.path.dirname(base_path)
    root_dir = os.path.dirname(base_dir)
    dataPath_ = os.path.join(root_dir, dataset_name, 'tasks', relation)
    featurePath = get_path_to_use_path(mode, relation)
    feature_stats = get_path_stats_path(mode, relation)

    relationId_path = os.path.join(root_dir, dataset_name, 'relation2id.txt')
    ent_id_path = os.path.join(root_dir, dataset_name, 'entity2id.txt')
    rel_id_path = os.path.join(root_dir, dataset_name, 'relation2id.txt')

    test_data_path = os.path.join(root_dir, dataset_name, 'tasks', relation, 'sort_test.pairs')
    log_file_name = os.path.join(base_dir, "fact_prediction_eval_results", mode, relation[8:] + ".txt")

    print(test_data_path)

    def bfs_two(e1, e2, path, kb, kb_inv):
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
                for entity in left:
                    try:
                        for path_ in kb.getPathsFrom(entity):
                            if path_.relation == left_step:
                                left_next.add(path_.connected_entity)
                    except Exception as e:
                        # print 'left', len(left)
                        print(('left', len(left)))
                        print(('left', len(left)), file=open(
                            log_file_name, encoding="utf-8", mode="at"))
                        # print left
                        print(left)
                        print(left, file=open(log_file_name,
                                              encoding="utf-8", mode="at"))
                        # print 'not such entity'
                        print('not such entity')
                        print('not such entity', file=open(
                            log_file_name, encoding="utf-8", mode="at"))
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
                        print(('right', len(right)))
                        print(('right', len(right)), file=open(
                            log_file_name, encoding="utf-8", mode="at"))
                        # print 'no such entity'
                        print('no such entity')
                        print('no such entity', file=open(
                            log_file_name, encoding="utf-8", mode="at"))
                        return False
                right = right_next

        if len(right & left) != 0:  # intersection of right:set and left:set
            return True
        return False

    def get_features():
        # 创建一个空字典用于存储路径频率信息
        stats = {}

        # 从文件 feature_stats 中读取路径频率信息
        f = open(feature_stats)
        path_freq = f.readlines()
        f.close()

        # 遍历每一行路径频率信息，提取路径和频率，并存储到 stats 字典中
        for line in path_freq:
            path = line.split('\t')[0]
            num = int(line.split('\t')[1])
            stats[path] = num

        # 获取路径频率的最大值
        max_freq = np.max(list(stats.values()))

        # 创建一个空字典用于存储关系到ID的映射
        relation2id = {}

        # 从文件 relationId_path 中读取关系到ID的映射信息
        f = open(relationId_path)
        content = f.readlines()
        f.close()

        # 遍历每一行关系到ID的映射信息，提取关系和ID，并存储到 relation2id 字典中
        for line in content:
            relation2id[line.split()[0]] = int(line.split()[1])

        # 创建两个空列表用于存储有效路径的ID列表和路径名称列表
        useful_paths = []
        named_paths = []

        # 从文件 featurePath 中读取路径信息
        f = open(featurePath)
        paths = f.readlines()
        f.close()

        # 遍历每一行路径信息
        for line in paths:
            path = line.rstrip()

            # 如果路径不在 stats 中，跳过当前路径
            if path not in stats:
                continue
            # 如果最大频率大于1且当前路径的频率小于2，跳过当前路径
            elif max_freq > 1 and stats[path] < 2:
                # 即使是强化学习代理找到的路径，如果路径频率过低（随机的），则不使用该路径。
                continue

            # 计算路径的长度（使用的关系的数量）
            length = len(path.split(' -> '))

            # 如果路径长度小于等于10，将路径的关系转换为ID并添加到相应的列表中
            if length <= 10:
                pathIndex = []
                pathName = []
                relations = path.split(' -> ')

                for rel in relations:
                    pathName.append(rel)
                    rel_id = relation2id[rel]
                    pathIndex.append(rel_id)

                # 将路径的关系ID列表添加到 useful_paths
                useful_paths.append(pathIndex)
                # 将路径的关系名称列表添加到 named_paths
                named_paths.append(pathName)

        # 打印使用的路径数量
        print(('How many paths used: ', len(useful_paths)))
        print(('How many paths used: ', len(useful_paths)),
              file=open(log_file_name, encoding="utf-8", mode="wt"))

        # 返回有效路径的关系ID列表和路径名称列表
        return useful_paths, named_paths  # 仅使用 named_paths.

    print("mode == ", mode)
    f1 = open(ent_id_path)
    f2 = open(rel_id_path)
    content1 = f1.readlines()
    content2 = f2.readlines()
    f1.close()
    f2.close()

    # 获取实体和关系的标识符
    entity2id = {}
    relation2id = {}
    for line in content1:
        entity2id[line.split()[0]] = int(line.split()[1])

    for line in content2:
        relation2id[line.split()[0]] = int(line.split()[1])

    # 加载实体和关系的向量表示
    ent_vec_E = np.loadtxt(dataPath_ + '/entity2vec.unif')
    rel_vec_E = np.loadtxt(dataPath_ + '/relation2vec.unif')
    rel = relation.replace("_", ":")
    relation_vec_E = rel_vec_E[relation2id[rel], :]

    ent_vec_R = np.loadtxt(dataPath_ + '/entity2vec.bern')
    rel_vec_R = np.loadtxt(dataPath_ + '/relation2vec.bern')
    M = np.loadtxt(dataPath_ + '/A.bern')
    M = M.reshape([-1, 50, 50])
    relation_vec_R = rel_vec_R[relation2id[rel], :]
    M_vec = M[relation2id[rel], :, :]

    _, named_paths = get_features()
    path_weights = []
    for path in named_paths:
        weight = 1.0/len(path)
        path_weights.append(weight)
    # path_weights: [1/len(path), 1/len(path), ...]
    path_weights = np.array(path_weights)
    kb = KB()
    kb_inv = KB()

    # 加载关系图信息
    f = open(dataPath_ + '/graph.txt')
    kb_lines = f.readlines()
    f.close()

    # 构建关系图
    for line in kb_lines:
        e1 = line.split()[0]
        rel = line.split()[1]
        e2 = line.split()[2]
        kb.addRelation(e1, rel, e2)
        kb_inv.addRelation(e2, rel, e1)

    # 加载测试数据
    f = open(test_data_path)
    test_data = f.readlines()
    f.close()
    test_pairs = []
    test_labels = []
    test_set = set()
    print("测试数据数量", len(test_data))

    # 读取测试样本和标签
    for line in test_data:
        e1 = line.split(',')[0].replace('thing$', '')
        # e1 = '/' + e1[0] + '/' + e1[2:]
        e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
        # e2 = '/' + e2[0] + '/' + e2[2:]
        # if (e1 not in kb.entities) or (e2 not in kb.entities):
        # continue
        test_pairs.append((e1, e2))
        label = 1 if line[-2] == '+' else 0
        test_labels.append(label)

    # 初始化评估得分列表
    scores_E = []
    scores_R = []
    scores_rl = []

    print(('How many queries: ', len(test_pairs)))
    print(('How many queries: ', len(test_pairs)), file=open(
        log_file_name, encoding="utf-8", mode="at"))
    # test_pairs: [(e1, e2), (e1, e2), ...]

    # 循环遍历测试样本
    for idx, sample in enumerate(test_pairs):
        # print(('Query No.%d of %d' % (idx, len(test_pairs))))
        # print(('Query No.%d of %d' % (idx, len(test_pairs))), file=open(log_file_name, encoding="utf-8", mode="at"))
        e1_vec_E = ent_vec_E[entity2id[sample[0]], :]
        e2_vec_E = ent_vec_E[entity2id[sample[1]], :]
        score_E = -np.sum(np.square(e1_vec_E + relation_vec_E - e2_vec_E))
        scores_E.append(score_E)

        e1_vec_R = ent_vec_R[entity2id[sample[0]], :]

        e2_vec_R = ent_vec_R[entity2id[sample[1]], :]
        e1_vec_rel = np.matmul(e1_vec_R, M_vec)
        e2_vec_rel = np.matmul(e2_vec_R, M_vec)
        score_R = -np.sum(np.square(e1_vec_rel + relation_vec_R - e2_vec_rel))
        scores_R.append(score_R)

        features = []
        for path in named_paths:
            features.append(
                int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
        # features = features*path_weights
        score_rl = sum(features)
        scores_rl.append(score_rl)

    # 对得分列表进行排序
    rank_stats_E = list(zip(scores_E, test_labels))
    rank_stats_R = list(zip(scores_R, test_labels))
    rank_stats_rl = list(zip(scores_rl, test_labels))
    # 按照 score_E 从高到低的顺序排序（e1_vec + rel - e2_vec）。虽然所有的值都是负数，但越接近零表示预测越准确。
    rank_stats_E.sort(key=lambda x: x[0], reverse=True)
    rank_stats_R.sort(key=lambda x: x[0], reverse=True)
    rank_stats_rl.sort(key=lambda x: x[0], reverse=True)

    def fact_prediction_on_RL(rank_stats_rl):
        correct = 0
        ranks = []
        for idx, item in enumerate(rank_stats_rl):
            if item[1] == 1:
                correct += 1
                ranks.append(correct/(1.0+idx))
        ap3 = np.mean(ranks)
        print(('RL: ', ap3))
        print(('RL: ', ap3), file=open(
            log_file_name, encoding="utf-8", mode="at"))
        return ap3

    def fact_prediction_on_TransE(rank_stats_E):
        correct = 0
        ranks = []
        for idx, item in enumerate(rank_stats_E):
            if item[1] == 1:
                correct += 1
                ranks.append(correct/(1.0+idx))
        ap1 = np.mean(ranks)
        print(('TransE: ', ap1))
        print(('TransE: ', ap1), file=open(
            log_file_name, encoding="utf-8", mode="at"))
        return ap1

    def fact_prediction_on_TransR(rank_stats_R):
        correct = 0
        ranks = []
        for idx, item in enumerate(rank_stats_R):
            if item[1] == 1:
                correct += 1
                ranks.append(correct/(1.0+idx))
        ap2 = np.mean(ranks)
        print(('TransR: ', ap2))
        print(('TransR: ', ap2), file=open(
            log_file_name, encoding="utf-8", mode="at"))
        return ap2

    def fact_prediction_on_TransH():

        f1 = open(ent_id_path)
        f2 = open(rel_id_path)
        content1 = f1.readlines()
        content2 = f2.readlines()
        f1.close()
        f2.close()

        entity2id = {}
        relation2id = {}
        for line in content1:
            entity2id[line.split()[0]] = int(line.split()[1])

        for line in content2:
            relation2id[line.split()[0]] = int(line.split()[1])

        ent_vec = np.loadtxt(dataPath_ + '/entity2vec.vec')
        rel_vec = np.loadtxt(dataPath_ + '/relation2vec.vec')
        M = np.loadtxt(dataPath_ + '/A.vec')
        M = M.reshape([rel_vec.shape[0], -1])

        f = open(test_data_path)
        test_data = f.readlines()
        f.close()
        test_pairs = []
        test_labels = []
        # queries = set()
        for line in test_data:
            e1 = line.split(',')[0].replace('thing$', '')
            # e1 = '/' + e1[0] + '/' + e1[2:]
            e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
            # e2 = '/' + e2[0] + '/' + e2[2:]
            test_pairs.append((e1, e2))
            label = 1 if line[-2] == '+' else 0
            test_labels.append(label)

        score_all = []
        rel = relation.replace("_", ":")
        d_r = np.expand_dims(rel_vec[relation2id[rel], :], 1)
        w_r = np.expand_dims(M[relation2id[rel], :], 1)

        for idx, sample in enumerate(test_pairs):
            # print 'query node: ', sample[0], idx
            h = np.expand_dims(ent_vec[entity2id[sample[0]], :], 1)
            t = np.expand_dims(ent_vec[entity2id[sample[1]], :], 1)

            h_ = h - np.matmul(w_r.transpose(), h)*w_r
            t_ = t - np.matmul(w_r.transpose(), t)*w_r

            score = -np.sum(np.square(h_ + d_r - t_))
            score_all.append(score)

        score_label = list(zip(score_all, test_labels))
        stats = sorted(score_label, key=lambda x: x[0], reverse=True)

        correct = 0
        ranks = []
        for idx, item in enumerate(stats):
            if item[1] == 1:
                correct += 1
                ranks.append(correct/(1.0+idx))
        ap4 = np.mean(ranks)
        # print 'TransH: ', ap4
        print(('TransH: ', ap4))
        print(('TransH: ', ap4), file=open(
            log_file_name, encoding="utf-8", mode="at"))
        return ap4

    def fact_prediction_on_TransD():
        ent_vec_D = np.loadtxt(dataPath_ + '/entity2vec.vec_D')
        rel_vec_D = np.loadtxt(dataPath_ + '/relation2vec.vec_D')
        M_D = np.loadtxt(dataPath_ + '/A.vec_D')
        ent_num = ent_vec_D.shape[0]
        rel_num = rel_vec_D.shape[0]
        rel_tran = M_D[0:rel_num, :]
        ent_tran = M_D[rel_num:, :]
        dim = ent_vec_D.shape[1]

        rel_id = relation2id[rel]
        r = np.expand_dims(rel_vec_D[rel_id, :], 1)
        r_p = np.expand_dims(rel_tran[rel_id, :], 1)
        scores_all_D = []
        for idx, sample in enumerate(test_pairs):
            h = np.expand_dims(ent_vec_D[entity2id[sample[0]], :], 1)
            h_p = np.expand_dims(ent_tran[entity2id[sample[0]], :], 1)
            t = np.expand_dims(ent_vec_D[entity2id[sample[1]], :], 1)
            t_p = np.expand_dims(ent_tran[entity2id[sample[1]], :], 1)
            M_rh = np.matmul(r_p, h_p.transpose()) + np.identity(dim)
            M_rt = np.matmul(r_p, t_p.transpose()) + np.identity(dim)
            score = - np.sum(np.square(M_rh.dot(h) + r - M_rt.dot(t)))
            scores_all_D.append(score)

        score_label = list(zip(scores_all_D, test_labels))
        stats = sorted(score_label, key=lambda x: x[0], reverse=True)

        correct = 0
        ranks = []
        for idx, item in enumerate(stats):
            if item[1] == 1:
                correct += 1
                ranks.append(correct/(1.0+idx))
        ap5 = np.mean(ranks)
        # print 'TransD: ', ap5
        print(('TransD: ', ap5))
        print(('TransD: ', ap5), file=open(
            log_file_name, encoding="utf-8", mode="at"))
        return ap5

    rl_score = fact_prediction_on_RL(rank_stats_rl)
    TransE_score = fact_prediction_on_TransE(rank_stats_E)
    TransR_score = fact_prediction_on_TransR(rank_stats_R)
    TransH_score = fact_prediction_on_TransH()
    TransD_score = fact_prediction_on_TransD()

    # # 写入结果到文件fact_prediction_eval_results/summary.txt
    # all_summary = f"relation:{relation}\n(RL:{rl_score}),(TransE:{TransE_score}),(TransR:{TransR_score}),(TransH:{TransH_score}),(TransD:{TransD_score})"
    # f = open(f"./fact_prediction_eval_results/{dataset_name}-summary.txt",
    #          encoding="utf-8", mode="at")
    # f.write(all_summary + "\n")
    # f.close()

    # 写入结果到文件 {base_dir}/results/{dataset_name}-fact_prediction_eval_results.csv
    result_file = os.path.join(base_dir, "results",
                               f"{dataset_name}-{mode}-fact_prediction_eval_results.csv")
    # 如果不存在，则创建文件，并追加表头。relation、RL、TransE、TransR、TransH、TransD
    if not os.path.exists(result_file):
        f = open(result_file, encoding="utf-8", mode="at")
        f.write(f"relation,{mode},TransE,TransR,TransH,TransD\n")
        f.close()
    # 追加一行数据
    f = open(result_file, encoding="utf-8", mode="at")
    f.write(
        f"{relation},{rl_score},{TransE_score},{TransR_score},{TransH_score},{TransD_score}\n")
    f.close()


if __name__ == "__main__":
    for relation in relations:
        print("relation == ", relation)
        fact_prediction_single(mode="AC", relation=relation)
