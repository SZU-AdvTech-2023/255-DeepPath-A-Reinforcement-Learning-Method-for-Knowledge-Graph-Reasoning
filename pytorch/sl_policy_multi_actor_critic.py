# python
from concurrent.futures import ProcessPoolExecutor
import torch
from torch.nn.functional import one_hot
from itertools import count
import sys
from setproctitle import setproctitle
# framework
from networks import PolicyNeuralNetTorch, Actor, Critic
from env import Env
from utils import *
import os
# 可视化
import datetime
import wandb
import torch.nn.functional as F



def multi_supervised_learning(relation, mode, extra_title ="",wandb_show=True,project_name ="deep-path"):
# def multi_supervised_learning(relation, wandb_show=True):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def train():
        # 从文件加载实体和关系的向量表示
        base_path = os.path.dirname(os.path.abspath(__file__))
        relation_path = os.path.join(dataPath, "tasks", relation, "train_pos")
        entity2vec_path = os.path.join(dataPath, "entity2vec.bern")
        relation2vec_path = os.path.join(dataPath, "relation2vec.bern")
        kb_env_rl_path = os.path.join(dataPath, "kb_env_rl.txt")
        # 保存模型和优化器的路径
        # save_model_path = os.path.join(
        #     base_path, 'torchmodels', 'policy_supervised_' + relation + ".pth")
        # save_optimizer_path = os.path.join(base_path, 'torchmodels', 'policy_supervised_' +
        #                                    relation + "_opt.pth")

        actor_model_path = os.path.join(
            base_path, 'torchmodels', 'supervised_actor_' + relation + ".pth")
        critic_model_path = os.path.join(
            base_path, 'torchmodels', 'supervised_critic_' + relation + ".pth")

        actor_optimizer_path = os.path.join(
            base_path, 'torchmodels', 'supervised_actor_optimizer_' + relation + "_opt.pth")
        critic_optimizer_path = os.path.join(
            base_path, 'torchmodels', 'supervised_critic_optimizer_' + relation + "_opt.pth")

        filter_relation = relation.split("_")[-1]
        if wandb_show:
            wandb.init(project=project_name,name=f'supervised_{mode}_{filter_relation}{extra_title}',config=hyperparameters)

        # 打开包含训练数据的文件
        f = open(relation_path)
        train_data = f.readlines()
        f.close()

        # 限制训练样本数量
        num_samples = min(len(train_data), max_num_samples)
        print("num_samples", num_samples)
        print("train_data size", len(train_data))

        # # 创建初始的监督训练模型并保存
        # model = PolicyNeuralNetTorch().to(device)
        # # 使用Adam优化器，学习率为learning_rate
        # optimizer = torch.optim.Adam(
        #     model.parameters(), lr=learning_rate)

        # 创建Actor-Critic模型和优化器
        # 创建actor模型和优化器
        actor = Actor().to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=learning_rate)
        # 创建critic模型和优化器
        critic = Critic().to(device)

        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=learning_rate)


        # 检查actor模型文件是否存在，如果存在则加载
        # if os.path.exists(actor_model_path):
        #     actor = torch.load(actor_model_path)
        #     actor_optimizer = torch.load(actor_optimizer_path)
        #     print("Loaded actor model and optimizer from:", actor_model_path)
        # else:
        #     print("No pre-trained actor model found.")

        # # 检查critic模型文件是否存在，如果存在则加载
        # if os.path.exists(critic_model_path):
        #     critic = torch.load(critic_model_path)
        #     critic_optimizer = torch.load(critic_optimizer_path)
        #     print("Loaded critic model and optimizer from:", critic_model_path)
        # else:
        #     print("No pre-trained critic model found.")


        # 从文件加载实体和关系的向量表示
        entity2vec = np.loadtxt(entity2vec_path)
        relation2vec = np.loadtxt(relation2vec_path)
        # print(dataPath + "entity2vec.bern")
        # print("entity2vec.shape", entity2vec.shape)
        # print("relation2vec.shape", relation2vec.shape)
        entity2id, relation2id, relations = open_entity2id_and_relation2id()

        # 从文件加载知识图谱环境
        with open(kb_env_rl_path) as f:
            kb_all = f.readlines()
        kb = relation_to_kb(relation)
        eliminated_kb = []
        concept = "concept:" + relation
        for line in kb_all:
            rel = line.split()[2]
            if rel != concept and rel != concept + "_inv":
                eliminated_kb.append(line)

        # 对于每个训练样本
        for episode in range(num_samples):
            # print(f"Episode {episode}")
            # print("Training Sample:", train_data[episode][:-1])
            if episode % 100 == 0:
                print(f"Episode {episode}")
                print("Training Sample:", train_data[episode][:-1])

            # 创建环境对象，用于强化学习训练
            env = Env(entity2vec, relation2vec, kb_all, eliminated_kb,
                      entity2id, relation2id, relations, train_data[episode])
            sample = train_data[episode].split()

            try:
                # 使用teacher函数生成好的强化学习训练数据
                # good_episodes = teacher(sample[0], sample[1], 5, env, kb)
                good_episodes = teacher(
                    sample[0], sample[1], 5, entity2vec, entity2id, relation2id, kb)
                # print("teacher生成好的强化学习训练数据数量", len(good_episodes))
            except Exception as e:
                print("Cannot find a path", e)
                continue

            # 对于每个好的训练数据
            for item in good_episodes:
                state_batch = []
                action_batch = []
                reward_batch = []  # Actor-Critic新增

                # 对于每一步的转换
                for t, transition in enumerate(item):
                    state_batch.append(transition.state)  # state: vector
                    action_batch.append(transition.action)  # action: ID
                    reward_batch.append(transition.reward)

                state_batch = np.squeeze(state_batch)
                state_batch = np.reshape(
                    state_batch, [-1, state_dim])  # (3, 200)
                state_batch = torch.tensor(state_batch, dtype=torch.float32).to(device)
                action_batch = torch.tensor(action_batch).to(device)  # (3,)
                reward_batch = torch.tensor(
                    reward_batch, dtype=torch.float).to(device)

                critic_value = critic(state_batch)

                advantage = reward_batch - critic_value
                # print("advantage", advantage)

                # Actor loss
                action_prob = actor.forward(state_batch)
                onehot_action = one_hot(
                    action_batch, torch.tensor(action_space).to(device))
                action_mask = onehot_action.type(torch.bool)
                picked_action_prob = torch.masked_select(
                    action_prob, action_mask)

                # 确保两个张量维度相同
                # print("value_batch shape:", value_batch.shape)
                # print("reward_batch shape:", reward_batch.shape)

                actor_loss = - \
                    torch.sum(torch.log(picked_action_prob) * advantage)

                # criterion = torch.nn.MSELoss()
                # critic_loss = criterion(value_batch, reward_batch)
                # Critic loss 使用 MSE Loss
                # print("-----------")        
                # print(critic_value.shape)
                # print(reward_batch.shape)
                # critic_loss = F.mse_loss(value_batch.squeeze(), reward_batch)
                critic_loss = F.mse_loss(critic_value.reshape(-1),reward_batch.reshape(-1).to(device))

                # critic_loss = F.smooth_l1_loss(
                #     value_batch.squeeze(), reward_batch)
                # critic_loss = F.smooth_l1_loss(value_batch, torch.squeeze(state_value))

                """
                平滑的 L1 损失函数，通常用于回归问题。在强化学习中，这个损失函数用于评估 Critic 网络的预测值与实际奖励之间的差异。
                让我解释一下每个参数的含义：
                value_batch.squeeze(): 这里对 Critic 网络的输出进行了压缩操作，即去除维度大小为1的维度。
                通常，Critic 网络的输出是一个表示状态值的张量，但它可能包含不必要的维度。通过使用 squeeze()，我们确保输出张量没有大小为1的维度，以匹配后续操作的期望形状。
                reward_batch: 这是实际的奖励值，通常由环境提供。在强化学习中，Critic 网络的目标是学习一个值函数，该值函数能够准确地预测在给定状态下获得的未来奖励的总和。
                F.smooth_l1_loss 将这两个张量作为输入，并计算它们之间的平滑的 L1 损失。这个损失衡量了 Critic 网络的输出与实际奖励之间的差异，帮助网络调整参数以提高预测准确性。
                这是训练 Critic 网络的一部分，以便更好地估计在不同状态下的累积奖励。
                """

                # 反向传播
                actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_optimizer.step()

                # Backpropagation 更新 Critic 模型
                critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                critic_optimizer.step()


                # 可视化
                if wandb_show:
                    wandb.log({"actor_loss": actor_loss.item(),
                              "critic_loss": critic_loss.item()})

        # 保存模型
        # torch.save(model, save_model_path)
        # torch.save(optimizer, save_optimizer_path)
        # 保存Actor模型和优化器
        torch.save(actor, actor_model_path)
        torch.save(actor_optimizer, actor_optimizer_path)

        # 保存Critic模型和优化器
        torch.save(critic, critic_model_path)
        torch.save(critic_optimizer, critic_optimizer_path)
        print('model saved')

    # print("relationPath", relationPath)

    max_num_samples = 500  # 原始代码: 500

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    train()

# 单进程测试
# if __name__ == "__main__":
#     setproctitle("supervised_actor_critic")
#     for relation in relations:
#         print("relation", relation)
#         multi_supervised_learning(relation,False)

# 多进程
if __name__ == "__main__":
    torch.set_printoptions(threshold=10000)  
    
    # 使用 ProcessPoolExecutor 开启多进程
    with ProcessPoolExecutor(max_workers=max(len(relations),6)) as executor:
        # 提交任务
        futures = []
        for relation in relations:
            print("Relation:", relation)
            future = executor.submit(multi_supervised_learning, relation)
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            future.result()
