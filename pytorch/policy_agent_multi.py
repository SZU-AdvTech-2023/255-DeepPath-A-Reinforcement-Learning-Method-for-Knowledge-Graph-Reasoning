from concurrent.futures import ProcessPoolExecutor
import datetime
import torch
from torch.nn.functional import one_hot
from torch import Tensor
import numpy as np
import collections
from itertools import count
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
from setproctitle import setproctitle

from utils import *
from env import Env
# 可视化
import wandb

import torch.multiprocessing as mp

# 设置启动方法为'spawn'
mp.set_start_method('spawn', force=True)

set_seed(seed_number)

device = torch.device(
    cuda) if torch.cuda.is_available() else torch.device("cpu")


def policy_train_single(relation, mode="RL", task="", log_save_path="",project_name="deep-path",wandb_show=False):
    # 改为os.path.join
    graphpath = os.path.join(dataPath, "tasks", relation, "graph.txt")
    relationPath = os.path.join(dataPath, "tasks", relation, "train_pos")
    entity2vec_path = os.path.join(dataPath, "entity2vec.bern")
    relation2vec_path = os.path.join(dataPath, "relation2vec.bern")
    kb_env_rl_path = os.path.join(dataPath, 'kb_env_rl.txt')
    base_path = os.path.dirname(__file__)

    # 模型和优化器路径
    supervised_model_path = os.path.join(
        base_path, "torchmodels",  "policy_supervised_" + relation + ".pth")
    supervised_optimizer_path = os.path.join(
        base_path, "torchmodels", "policy_supervised_" + relation + "_opt.pth")
    retrained_model_path = os.path.join(
        base_path, "torchmodels", "policy_retrained_" + relation + ".pth")
    retrained_optimizer_path = os.path.join(
        base_path, "torchmodels", "policy_retrained_" + relation + "_opt.pth")

    max_episode_num = 600  # original code : 300
    max_test_num = 500  # original code : 500

    def predict(model: torch.nn.Module, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float).to(device)
        return model(state).detach().cpu().numpy()

    def update(model: torch.nn.Module,
               state: np.ndarray,
               target,
               action,
               optimizer: torch.optim.Optimizer) -> Tensor:
        # not same code with sl_policy.py! target is applied.
        state = torch.tensor(state, dtype=torch.float).to(device)
        # creating tensor from a list of ndarrays is slow?
        action = torch.tensor(action).to(device)
        action_prob = model.forward(state)
        # print(action)
        # print(torch.tensor(action_space))
        onehot_action = one_hot(action.to(torch.int64),
                                torch.tensor(action_space))
        action_mask = onehot_action.type(torch.bool)
        # print("---")
        # print(action.size())
        # print(action_prob.size())
        # print(action_mask.size())
        # print("---")
        picked_action_prob = torch.masked_select(action_prob, action_mask)
        loss = torch.sum(-torch.log(picked_action_prob) * target)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def REINFORCE(training_pairs, model, optimizer, num_episodes):
        # 删除relation前面的concept_(若有)
        filter_relation = relation.split("_")[-1]
        print("Start RL training", filter_relation)
        if wandb_show:
            wandb.init(project=project_name, name=f"Training_{mode}_{filter_relation}",config=hyperparameters)

        train = training_pairs
        success = 0
        path_found_entity = []  # 存储找到的路径
        path_relation_found = []  # 存储路径的关系

        entity2id, relation2id, relations = open_entity2id_and_relation2id()

        entity2vec = np.loadtxt(entity2vec_path)
        relation2vec = np.loadtxt(relation2vec_path)

        with open(kb_env_rl_path) as f:
            kb_all = f.readlines()

        kb = relation_to_kb(relation)
        eliminated_kb = []
        concept = "concept:" + relation

        # 过滤掉与关系无关的知识图谱三元组
        for line in kb_all:
            rel = line.split()[2]
            if rel != concept and rel != concept + "_inv":
                eliminated_kb.append(line)

        all_reward = 0
        all_length = 0
        # 对每个episode进行训练
        for i_episode in range(num_episodes):
            start = time.time()
            print(f"Episode {i_episode}")
            print("Training sample: ", train[i_episode][:-1])

            env = Env(entity2vec, relation2vec, kb_all, eliminated_kb,
                      entity2id, relation2id, relations, train[i_episode])

            sample = train[i_episode].split()
            state_idx = [env.entity2id_[sample[0]],
                         env.entity2id_[sample[1]], 0]

            episode = []
            state_batch_negative = []
            action_batch_negative = []

            for t in count():
                state_vec = idx_state(entity2vec, state_idx)

                # 根据当前状态选择动作
                action_probs = predict(model, state_vec)
                action_chosen = np.random.choice(
                    np.arange(action_space), p=np.squeeze(action_probs))

                # 执行动作并获取奖励
                reward, new_state, done = env.interact(
                    state_idx, action_chosen)

                # 如果动作失败，将其记录下来以后进行惩罚
                if reward == -1:
                    state_batch_negative.append(state_vec)
                    action_batch_negative.append(action_chosen)

                new_state_vec = idx_state(entity2vec, new_state)
                episode.append(Transition(
                    state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

                if done or t == max_steps:
                    break

                state_idx = new_state

            # 当存在无效步骤时，对智能体进行惩罚
            if len(state_batch_negative) != 0:
                print("Penalty to invalid steps:", len(state_batch_negative))
                loss = update(model, np.reshape(state_batch_negative, (-1, state_dim)), -
                       0.05, action_batch_negative, optimizer)  # 对无效步骤进行惩罚
                wandb.log({
                    'Loss': loss.item(),
                })
            print("----- FINAL PATH -----")
            print("\t".join(env.path))
            print("PATH LENGTH", len(env.path))
            print("----- FINAL PATH -----")
            # 如果成功，进行一次优化
            if done == 1:
                print("Success")

                path_found_entity.append(path_clean(" -> ".join(env.path)))

                success += 1
                path_length = len(env.path)
                length_reward = 1 / path_length
                global_reward = 1
                # 这里应该还要计算余弦相似度得分
                # if len(path_found) != 0:
                # 	path_found_embedding = [env.path_embedding(path.split(' -> ')) for path in path_found]
                # 	curr_path_embedding = env.path_embedding(env.path_relations)
                # 	path_found_embedding = np.reshape(path_found_embedding, (-1,embedding_dim))
                # 	cos_sim = cosine_similarity(path_found_embedding, curr_path_embedding)
                # 	diverse_reward = -np.mean(cos_sim)
                # 	print 'diverse_reward', diverse_reward
                # 	total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward
                # else:
                # 	total_reward = 0.1*global_reward + 0.9*length_reward
                # path_found.add(' -> '.join(env.path_relations))

                total_reward = 0.1 * global_reward + 0.9 * length_reward
                state_batch = []
                action_batch = []

                for t, transition in enumerate(episode):
                    if transition.reward == 0:
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)
                all_reward += total_reward
                all_length += len(env.path)

                # 更新模型参数
                loss = update(model, np.reshape(state_batch, (-1, state_dim)),
                       total_reward, action_batch, optimizer)
                if wandb_show:
                    wandb.log({
                        'Retrain Loss': loss.item(),
                    })
            else:
                global_reward = -0.05
                total_reward = global_reward
                state_batch = []
                action_batch = []
                all_reward += total_reward
                for t, transition in enumerate(episode):
                    if transition.reward == 0:
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)

                # 更新模型参数
                loss = update(model, np.reshape(state_batch, (-1, state_dim)),
                       total_reward, action_batch, optimizer)
                if wandb_show:
                    wandb.log({
                        'Retrain Loss': loss.item(),
                    })

                # wandb.log({
                #     'Loss': loss.item(),
                #     "fail_loss":loss.item()
                # })
                print("Failed, Do one teacher guidline")
                try:
                    # 使用老师模型提供的示例进行优化
                    # good_episodes = teacher(sample[0], sample[1], 1, env, kb)
                    good_episodes = teacher(sample[0], sample[1], 1, entity2vec, entity2id, relation2id, kb)

                    for item in good_episodes:
                        teacher_state_batch = []
                        teacher_action_batch = []
                        total_reward = 0.0 * 1 + 1 * 1 / len(item)
                        for t, transition in enumerate(item):
                            teacher_state_batch.append(transition.state)
                            teacher_action_batch.append(transition.action)
                        # 更新模型参数
                        loss = update(model, np.squeeze(teacher_state_batch),
                               1, teacher_action_batch, optimizer)
                        # wandb.log({
                        #   'Loss': loss.item(),
                        #   "teacher_loss":loss.item()
                        # })
                    print("teacher guideline success")
                except Exception as e:
                    print("Teacher guidline failed")

            print("Episode time: ", time.time() - start, "\n")
            # 记录关键指标到WandB
            if wandb_show:
                wandb.log({
                    'Success Percentage': success / num_episodes,
                    'Average Path Length': all_length / (success + 1e-5),
                    'All Reward': all_reward,
                    "Retrain Loss": loss
                })
        if wandb_show:
            wandb.finish()

        print("Success percentage:", success / num_episodes)

        for path in path_found_entity:
            rel_ent = path.split(" -> ")
            path_relation = []
            for idx, item in enumerate(rel_ent):
                if idx % 2 == 0:
                    path_relation.append(item)
            path_relation_found.append(" -> ".join(path_relation))

        relation_path_stats = list(
            collections.Counter(path_relation_found).items())
        relation_path_stats = sorted(
            relation_path_stats, key=lambda x: x[1], reverse=True)

        # 将路径统计保存到文件中
        with open(get_path_stats_path(mode, relation), "w") as f:
            for item in relation_path_stats:
                f.write(item[0] + "\t" + str(item[1]) + "\n")
        print("Path stats saved")

        return

    def retrain():
        print("Start retraining")
        with open(relationPath) as f:
            training_pairs = f.readlines()
        print("Training data loaded")
        model = torch.load(supervised_model_path)
        # implements L2 regularization by weight_decay (TODO needs verification)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # TODO deleted weight_decay, load optimizer
        optimizer = torch.load(supervised_optimizer_path)
        print("sl_policy restored")

        episodes = min(len(training_pairs), max_episode_num)
        print("Training episodes:", episodes)
        REINFORCE(training_pairs, model, optimizer, episodes)
        torch.save(model, retrained_model_path)
        torch.save(optimizer, retrained_optimizer_path)
        print("Retrained model saved")

    def test():
        # 读取实体和关系的标识符，以及关系列表
        entity2id, relation2id, relations = open_entity2id_and_relation2id()


        filter_relation = relation.split("_")[-1]
        print(f"Start {mode} training {filter_relation}")
        if wandb_show:
            wandb.init(project=project_name, name=f"TEST_{mode}_{filter_relation}",config=hyperparameters)

        # 在日志文件中记录当前时间
        with open(log_save_path, "wt") as f:
            current_time = datetime.datetime.now()
            f.write("time:{current_time}\n")

        # 读取测试数据
        with open(relationPath) as f:
            test_data = f.readlines()
        test_num = min(len(test_data), max_test_num)

        success = 0

        path_found = list()
        path_relation_found = list()
        path_set = set()

        # 加载预训练的模型和优化器
        model = torch.load(retrained_model_path)
        optimizer = torch.load(retrained_optimizer_path)
        print("Model reloaded")

        # 加载实体和关系的向量表示
        entity2vec = np.loadtxt(entity2vec_path)
        relation2vec = np.loadtxt(relation2vec_path)

        # 读取知识库的三元组信息
        with open(kb_env_rl_path) as f:
            kb_all = f.readlines()
        eliminated_kb = []

        # 从知识库中排除与当前测试关系不相关的三元组
        concept = "concept:" + relation
        for line in kb_all:
            rel = line.split()[2]
            if rel != concept and rel != concept + "_inv":
                eliminated_kb.append(line)

        # 循环遍历测试样本
        for episode in range(test_num):
            # print(f"Test sample {episode}: {test_data[episode][:-1]}")
            with open(log_save_path, mode="at", encoding="utf-8") as f:
                f.write(
                    f"\nTest sample {episode}: {test_data[episode][:-1]}\n")

            # 创建环境对象
            env = Env(entity2vec, relation2vec, kb_all, eliminated_kb,
                      entity2id, relation2id, relations, test_data[episode])
            sample = test_data[episode].split()

            # 获取初始状态索引
            state_idx = [env.entity2id_[sample[0]],
                         env.entity2id_[sample[1]], 0]

            transitions = list()

            for t in count():
                # 获取当前状态的向量表示
                state_vec = idx_state(env.entity2vec, state_idx)

                # 使用模型进行动作预测
                action_probs = predict(model, state_vec)
                action_probs = np.squeeze(action_probs)

                # 在动作概率的基础上选择动作
                action_chosen = np.random.choice(
                    np.arange(action_space), p=action_probs)

                # 与环境交互，获取奖励和新的状态
                reward, new_state, done = env.interact(
                    state_idx, action_chosen)

                # 获取新状态的向量表示
                new_state_vec = idx_state(env.entity2vec, new_state)

                # 将状态转换信息添加到列表中
                transitions.append(Transition(
                    state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

                # 判断是否结束当前测试样本
                if done or t == max_steps_test:
                    if done:
                        success += 1
                        print("Success\n")
                        with open(log_save_path, mode="at", encoding="utf-8") as f:
                            f.write("Success\n")
                        path = path_clean(" -> ".join(env.path))
                        path_found.append(path)
                    else:
                        print("Episode ends due to step limit\n")
                        with open(log_save_path, mode="at", encoding="utf-8") as f:
                            f.write("Episode ends due to step limit\n")
                    break
                state_idx = new_state

            if done:
                if len(path_set) != 0:
                    # 计算多样性奖励
                    path_found_embedding = [env.path_embedding(
                        path.split(" -> "), embedding_dim) for path in path_set]
                    curr_path_embedding = env.path_embedding(
                        env.path_relations, embedding_dim)
                    path_found_embedding = np.reshape(
                        path_found_embedding, (-1, embedding_dim))
                    cos_sim = cosine_similarity(
                        path_found_embedding, curr_path_embedding)
                    diverse_reward = -np.mean(cos_sim)
                    # print("diverse_reward", diverse_reward)
                    state_batch = []
                    action_batch = []

                    # 为无奖励的状态收集训练数据
                    for t, transition in enumerate(transitions):
                        if transition.reward == 0:
                            state_batch.append(transition.state)
                            action_batch.append(transition.action)

                    # 更新模型
                    update(model, np.reshape(state_batch, (-1, state_dim)),
                           0.1*diverse_reward, action_batch, optimizer)

                path_set.add(" -> ".join(env.path_relations))

            if wandb_show:
                wandb.log({
                    'TEST Success Percentage': success / test_num,
                })
        
        if wandb_show:
            wandb.finish()

        for path in path_found:
            rel_ent = path.split(" -> ")
            path_relation = []
            for idx, item in enumerate(rel_ent):
                if idx % 2 == 0:
                    path_relation.append(item)
            path_relation_found.append(" -> ".join(path_relation))

        # 统计路径出现次数
        relation_path_stats = list(
            collections.Counter(path_relation_found).items())
        relation_path_stats = sorted(
            relation_path_stats, key=lambda x: x[1], reverse=True)

        ranking_path = []

        # 获取路径和长度的信息
        for item in relation_path_stats:
            path = item[0]
            length = len(path.split(" -> "))
            ranking_path.append((path, length))

        ranking_path = sorted(ranking_path, key=lambda x: x[1])
        print("Success percentage:", success/test_num)
        with open(log_save_path, mode="at", encoding="utf-8") as f:
            f.write("Success percentage: " + str(success/test_num) + "\n")

        # 保存路径到文件
        path_to_use_path = os.path.join(
            dataPath, "tasks", relation, "path_to_use.txt")
        with open(path_to_use_path, "w") as f:
            for item in ranking_path:
                f.write(item[0] + "\n")
        print("path to use saved")
        return success/test_num

    if task == 'test':
        return test()
    elif task == 'retrain':
        retrain()
    else:
        retrain()
        return test()


if __name__ == "__main__":
    torch.set_printoptions(threshold=10000)  
    task = ''
    mode = "RL"
    
    # 使用 ProcessPoolExecutor 开启多进程
    with ProcessPoolExecutor(max_workers=min(len(relations),6)) as executor:
        # 提交任务
        futures = []
        for relation in relations:
            print("Relation:", relation)
            log_save_path = f"policy_agent_{mode}_testlog_{relation}.txt"
            log_save_path = os.path.join(os.path.dirname(__file__), "log", log_save_path)
            
            future = executor.submit(policy_train_single, relation, mode, task, log_save_path)
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            future.result()
