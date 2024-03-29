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

from networks import Actor, Critic
import torch.nn.functional as F
# PPO相关
from torch.distributions import Categorical


device = torch.device(
    cuda) if torch.cuda.is_available() else torch.device("cpu")


def policy_train_single(relation, mode="RL", task="test", log_save_path=""):
    # 改为os.path.join
    graphpath = os.path.join(dataPath, "tasks", relation, "graph.txt")
    relationPath = os.path.join(dataPath, "tasks", relation, "train_pos")
    entity2vec_path = os.path.join(dataPath, "entity2vec.bern")
    relation2vec_path = os.path.join(dataPath, "relation2vec.bern")
    kb_env_rl_path = os.path.join(dataPath, 'kb_env_rl.txt')
    base_path = os.path.dirname(__file__)

    max_episode_num = 300  # original code : 300
    max_test_num = 500  # original code : 500



    # PPO算法的Actor模型和优化器路径
    # agent only的PPO算法
    supervised_actor = os.path.join(
        base_path, 'torchmodels', 'supervised_actor_' + relation + ".pth")
    supervised_actor_optimizer = os.path.join(
        base_path, 'torchmodels', 'supervised_actor_optimizer_' + relation + "_opt.pth")
    actor_model_path = os.path.join(
        base_path, 'torchmodels', 'ppo_actor_' + relation + ".pth")
    actor_optimizer_path = os.path.join(
        base_path, 'torchmodels', 'ppo_actor_optimizer_' + relation + "_opt.pth")

    # add Critic
    critic_model = Critic().to(device)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=learning_rate)


    # def predict(model: torch.nn.Module, state: np.ndarray) -> np.ndarray:
    #     state = torch.tensor(state, dtype=torch.float).to(device)
    #     model = model.to(device)  # 将模型移动到相同的设备上
    #     with torch.no_grad():
    #         action_probs = model(state)
    #         m = Categorical(action_probs)
    #         return m.sample().detach().cpu().numpy()

    def predict(model: torch.nn.Module, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float).to(device)
        model = model.to(device)  # 将模型移动到相同的设备上
        with torch.no_grad():
            return model(state).detach().cpu().numpy()



    def update(model: torch.nn.Module,
           state: np.ndarray,
           target,
           action,
           optimizer: torch.optim.Optimizer) -> Tensor:
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action).to(device)
        action_prob = model(state)
        m = Categorical(action_prob)
        log_prob = m.log_prob(action)
        ratio = torch.exp(log_prob - torch.log(torch.tensor(target, dtype=torch.float).to(device)))
        surr1 = ratio * torch.tensor(target, dtype=torch.float).to(device)
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * torch.tensor(target, dtype=torch.float).to(device)
        loss = -torch.min(surr1, surr2)

        # 计算平均损失
        loss = torch.mean(loss)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def PPO_train(training_pairs, actor_model, actor_optimizer, num_episodes,gap=10):
        # 删除relation前面的concept_(若有)
        filter_relation = relation.split("_")[-1]

        # 输出当前时间:年月日时分秒
        print("Start RL training", filter_relation,datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        wandb.init(project="deep-path", name="RL_PPO_Training_" + filter_relation)

        train = training_pairs
        success = 0
        path_found_entity = []  # 存储找到的路径
        path_relation_found = []  # 存储路径的关系
        all_reward = 0
        all_length = 0

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

        # 对每个episode进行训练
        for i_episode in range(num_episodes):
            start = time.time()
            if i_episode % gap == 0:
                print(f"Episode {i_episode}")
                # print("Training sample: ", train[i_episode][:-1])

            env = Env(entity2vec, relation2vec, kb_all, eliminated_kb,
                    entity2id, relation2id, relations, train[i_episode])

            sample = train[i_episode].split()
            state_idx = [env.entity2id_[sample[0]],
                        env.entity2id_[sample[1]], 0]

            episode = []
            state_batch_negative = []
            action_batch_negative = []

            # 不断寻找路径
            # t是episode的长度，即路径的长度。从0开始计数，最大值为max_steps
            for t in count():
                # print("t:", t)
                # print("state_idx:", state_idx)

                state_vec = idx_state(entity2vec, state_idx)
                # print("state_vec:", state_vec)
                action_probs = predict(actor_model, state_vec)
                # print("action_probs:", action_probs)

                action_chosen = np.random.choice(
                    np.arange(action_space), p=np.squeeze(action_probs))

                reward, new_state, done = env.interact(
                    state_idx, action_chosen)

                if reward == -1:
                    state_batch_negative.append(state_vec)
                    action_batch_negative.append(action_chosen)

                new_state_vec = idx_state(entity2vec, new_state)
                episode.append(Transition(
                    state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

                if done or t == max_steps:
                    break

                state_idx = new_state

            if len(state_batch_negative) != 0:
                if i_episode % gap == 0:
                    print("Penalty to invalid steps:", len(state_batch_negative))
                update(actor_model, np.reshape(state_batch_negative, (-1, state_dim)), -
                    0.05, action_batch_negative, actor_optimizer)
            if i_episode % gap == 0:
                print("----- FINAL PATH -----")
                print("\t".join(env.path))
                print("PATH LENGTH", len(env.path))
                print("----- FINAL PATH -----")

            # 如果找到了路径
            if done == 1:
                if i_episode % gap == 0:
                    print("Success")

                path_found_entity.append(path_clean(" -> ".join(env.path)))

                success += 1
                path_length = len(env.path)
                length_reward = 1 / path_length
                global_reward = 1
                all_length += len(env.path)

                total_reward = 0.1 * global_reward + 0.9 * length_reward
                all_reward += total_reward
                state_batch = []
                action_batch = []

                for t, transition in enumerate(episode):
                    if transition.reward == 0:
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)

                # 更新Actor模型参数
                update(actor_model, np.reshape(state_batch, (-1, state_dim)),
                    total_reward, action_batch, actor_optimizer)

                # 计算Critic的目标值
                state_values = critic_model(torch.tensor(np.vstack(state_batch), dtype=torch.float).to(device))
                target_values = torch.tensor([total_reward] * len(state_batch), dtype=torch.float).to(device)

                # 计算Critic的损失
                critic_loss = F.smooth_l1_loss(state_values.squeeze(), target_values)

                # 更新Critic模型参数
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            else:
                global_reward = -0.05
                total_reward = global_reward
                all_reward += total_reward
                state_batch = []
                action_batch = []

                for t, transition in enumerate(episode):
                    if transition.reward == 0:
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)

                update(actor_model, np.reshape(state_batch, (-1, state_dim)),
                    total_reward, action_batch, actor_optimizer)
                
                # 更新 Critic 模型参数
                critic_target_values = [total_reward] * len(state_batch)
                critic_model.critic_update(state_batch, critic_target_values, critic_optimizer)

                if i_episode % gap == 0:
                    print("Failed, Do one teacher guideline")
                try:
                    good_episodes = teacher(sample[0], sample[1], 1, env, kb)
                    for item in good_episodes:
                        teacher_state_batch = []
                        teacher_action_batch = []
                        total_reward = 0.0 * 1 + 1 * 1 / len(item)
                        for t, transition in enumerate(item):
                            teacher_state_batch.append(transition.state)
                            teacher_action_batch.append(transition.action)
                        
                        # 更新 Actor 模型参数
                        update(actor_model, np.squeeze(teacher_state_batch),
                            1, teacher_action_batch, actor_optimizer)

                        # 更新 Critic 模型参数
                        critic_target_values = [total_reward] * len(teacher_state_batch)
                        critic_model.critic_update(teacher_state_batch, critic_target_values, critic_optimizer)
                    if i_episode % gap == 0:
                        print("teacher guideline success")
                except Exception as e:
                    if i_episode % gap == 0:
                        print("Teacher guideline failed")
            # 每隔一段时间才输出
            if i_episode % gap == 0:
                print("Episode time: ", time.time() - start, "\n")
                print("all_length: ", all_length, "\n")
                print("success: ", success, "\n")
            
            wandb.log({
                'Success Percentage': success / num_episodes,
                'Average Path Length': all_length / (success + 0.001),
                'All Reward': all_reward / num_episodes,
                # 'Episode Time': time.time() - start
            })

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

        # 加载PPO Actor模型和优化器
        actor_model = torch.load(supervised_actor)
        actor_optimizer = torch.load(supervised_actor_optimizer)
        print("Actor model and optimizer restored")


        episodes = min(len(training_pairs), max_episode_num)
        print("Training episodes:", episodes)

        # 调用Actor-Critic训练函数
        PPO_train(training_pairs, actor_model,actor_optimizer, episodes)

        # 保存Actor和Critic模型和优化器
        torch.save(actor_model, actor_model_path)
        torch.save(actor_optimizer, actor_optimizer_path)

        print("Retrained Actor-Critic model saved")

    def test():
        # 读取实体和关系的标识符，以及关系列表
        entity2id, relation2id, relations = open_entity2id_and_relation2id()

        filter_relation = relation.split("_")[-1]
        print("Start RL training", filter_relation)

        wandb.init(project="deep-path", name="RL_PPO_TEST_" + filter_relation)

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

        # 加载预训练的 PPO agent-only 模型和优化器
        actor_model = torch.load(actor_model_path).to(device)
        actor_optimizer = torch.load(actor_optimizer_path)

        def set_requires_grad(model, requires_grad=True):
            for param in model.parameters():
                param.requires_grad = requires_grad

        # 更新模型
        set_requires_grad(actor_model, requires_grad=True)

        print("PPO agent-only Model reloaded")

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
        # 循环遍历测试样本
        for episode in range(test_num):
            print(f"Test sample {episode}: {test_data[episode][:-1]}")
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

                # 使用 Actor 模型进行动作预测
                action_probs = predict(actor_model, state_vec)
                action_probs = np.squeeze(action_probs)

                # 在动作概率的基础上选择动作
                action_chosen = np.random.choice(
                    np.arange(action_space), p=action_probs)

                # 使用 Critic 模型进行状态值预测
                # value = critic_model(torch.tensor(state_vec, dtype=torch.float).to(device))

                # 与环境交互，获取奖励和新的状态
                reward, new_state, done = env.interact(
                    state_idx, action_chosen)

                # 获取新状态的向量表示
                new_state_vec = idx_state(env.entity2vec, new_state)

                # 将状态转换信息添加到列表中
                # transitions.append(Transition(
                #     state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward, value=value))
                # 将状态转换信息添加到列表中
                # transitions.append(Transition(
                #     state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

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
                    print("diverse_reward", diverse_reward)
                    state_batch = []
                    action_batch = []

                    # 为无奖励的状态收集训练数据
                    for t, transition in enumerate(transitions):
                        if transition.reward == 0:
                            state_batch.append(transition.state)
                            action_batch.append(transition.action)


                    # 使用 PPO 算法计算 Actor Loss
                    # advantage = reward  # 在 PPO 中，优势是直接使用奖励
                    # actor-critic里面用critic输出计算优势
                    # 使用 Critic 输出计算优势
                    state_values = critic_model(torch.tensor(np.vstack(state_batch), dtype=torch.float).to(device))
                    advantage = reward - state_values.squeeze().detach().cpu().numpy()

                    action_probs = predict(actor_model, state_vec)
                    m = Categorical(torch.tensor(action_probs, dtype=torch.float).to(device))
                    old_log_prob = m.log_prob(torch.tensor(action_chosen).to(device))
                    ratio = torch.exp(m.log_prob(torch.tensor(action_chosen).to(device)) - old_log_prob)
                    surr1 = ratio * torch.tensor(advantage, dtype=torch.float).to(device)
                    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * torch.tensor(advantage, dtype=torch.float).to(device)
                    actor_loss = -torch.min(surr1, surr2)
                    actor_loss = torch.tensor(actor_loss, requires_grad=True)

                    # 使用 PPO 算法更新 Actor 模型
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # target_values = torch.tensor([total_reward] * len(state_batch), dtype=torch.float).to(device)

                    # # 计算Critic的损失
                    # critic_loss = F.smooth_l1_loss(state_values, target_values)

                    # # 更新Critic模型参数
                    # critic_optimizer.zero_grad()
                    # critic_loss.backward()
                    # critic_optimizer.step()


                    # # 计算 Actor Loss
                    # actor_loss = -torch.log(action_probs[action_chosen]) * advantage.item()

                    # # Backpropagation 更新 Actor 模型
                    # actor_optimizer.zero_grad()
                    # actor_loss.backward()
                    # actor_optimizer.step()

                path_set.add(" -> ".join(env.path_relations))

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
        
        path_to_use_path = get_path_to_use_path(mode, relation)
        with open(path_to_use_path, "w") as f:
            for item in ranking_path:
                f.write(item[0] + "\n")
        print("path to use saved")
        return

    if task == 'test':
        test()
    elif task == 'retrain':
        retrain()
    else:
        retrain()
        test()


if __name__ == "__main__":
    torch.set_printoptions(threshold=10000)  
    task = ''
    mode = "PPO"
    # 遍历所有的关系
    for relation in relations:
        print("Relation:", relation)
        log_save_path = f"policy_agent_{mode}_testlog_{relation}.txt"
        log_save_path = os.path.join(
            os.path.dirname(__file__), "log", log_save_path)
        policy_train_single(relation=relation, mode=mode,
                            task=task, log_save_path=log_save_path)
