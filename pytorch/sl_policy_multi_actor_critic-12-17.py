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

# trick: 状态正则化
from util_normalization import Normalization

def multi_supervised_learning(relation, mode, extra_title ="",wandb_show=True,project_name ="deep-path"):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def train():
        # 从文件加载实体和关系的向量表示
        base_path = os.path.dirname(os.path.abspath(__file__))
        relation_path = os.path.join(dataPath, "tasks", relation, "train_pos")
        entity2vec_path = os.path.join(dataPath, "entity2vec.bern")
        relation2vec_path = os.path.join(dataPath, "relation2vec.bern")
        kb_env_rl_path = os.path.join(dataPath, "kb_env_rl.txt")
        # 保存模型和优化器的路径
        # actor_model_path = os.path.join(
        #     base_path, 'torchmodels', 'supervised_actor_' + relation + ".pth")
        # critic_model_path = os.path.join(
        #     base_path, 'torchmodels', 'supervised_critic_' + relation + ".pth")

        # actor_optimizer_path = os.path.join(
        #     base_path, 'torchmodels', 'supervised_actor_optimizer_' + relation + "_opt.pth")
        # critic_optimizer_path = os.path.join(
        #     base_path, 'torchmodels', 'supervised_critic_optimizer_' + relation + "_opt.pth")


        actor_checkpoint_path = os.path.join(
            base_path, 'torchmodels', 'supervised_actor_checkpoint_' + relation + ".pth")
        critic_checkpoint_path = os.path.join(
            base_path, 'torchmodels', 'supervised_critic_checkpoint' + relation + ".pth")

        # actor_optimizer_path = os.path.join(
        #     base_path, 'torchmodels', 'supervised_actor_optimizer_' + relation + "_opt.pth")
        # critic_optimizer_path = os.path.join(
        #     base_path, 'torchmodels', 'supervised_critic_optimizer_' + relation + "_opt.pth")

        
        if use_state_norm:
            state_norm = Normalization(state_dim)

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

        # trick: pytorch中Adam优化器默认的eps=1e-8，它的作用是提高数值稳定性
        # 在Open AI的baseline，和MAPPO的论文里，都单独设置eps=1e-5，这个特殊的设置可以在一定程度上提升算法的训练性能。


        # 创建Actor-Critic模型和优化器
        # 创建actor模型和优化器
        actor = Actor().to(device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=learning_rate,eps=custom_eps)
        # 创建critic模型和优化器
        critic = Critic().to(device)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=learning_rate,eps=custom_eps)


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
        good_nums = 0
        # 对于每个训练样本
        for episode in range(num_samples):
            # print(f"Episode {episode}")
            # print("Training Sample:", train_data[episode][:-1])
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
                # print(f"生成{len(good_episodes)}条专家路径")
                good_nums += len(good_episodes)
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

                actor_loss =  torch.sum( -torch.log(picked_action_prob) * advantage)

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
                # 平滑的 L1 损失函数，用于评估 Critic 网络的预测值与实际奖励之间的差异。
                # critic_loss = F.smooth_l1_loss(value_batch, torch.squeeze(state_value))

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
                    wandb.log({"sl_actor_loss": actor_loss.item(),
                              "sl_critic_loss": critic_loss.item()})


        print(f"一共生成{good_nums}条专家路径")

        # 保存Actor模型和优化器
        actor_checkpoint = {
            'state_dict': actor.state_dict(),
            'optimizer': actor_optimizer.state_dict()
        }
        critic_checkpoint = {
            'state_dict': critic.state_dict(),
            'optimizer': critic_optimizer.state_dict()
        }        
        # 保存模型和优化器
        torch.save(actor_checkpoint, actor_checkpoint_path)
        torch.save(critic_checkpoint, critic_checkpoint_path)
        print('model saved')
        if wandb_show:
            wandb.save(actor_checkpoint_path)
            wandb.save(critic_checkpoint_path)
            wandb.finish()
    # print("relationPath", relationPath)

    max_num_samples = 500  # 原始代码: 500

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    train()





# 单进程测试
# if __name__ == "__main__":
#     setproctitle("supervised_actor_critic")
#     for relation in relations[:1]:
#         print("relation", relation)
#         multi_supervised_learning(relation,True)

# 多进程
if __name__ == "__main__":
    torch.set_printoptions(threshold=10000)  
    mode = "AC"
    wandb_show = False
    extra_title = ""
    # 使用 ProcessPoolExecutor 开启多进程
    with ProcessPoolExecutor(max_workers=min(len(relations),6)) as executor:
        # 提交任务
        futures = []
        for relation in relations:
            print("Relation:", relation)
            future = executor.submit(multi_supervised_learning, relation,mode,extra_title,wandb_show)
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            future.result()
