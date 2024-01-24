# python
from concurrent.futures import ProcessPoolExecutor
import torch
from torch.nn.functional import one_hot
from itertools import count
import sys
from setproctitle import setproctitle
# framework
from networks import PolicyNeuralNetTorch
from env import Env
from utils import *
import os
# 可视化
import datetime
import wandb

# trick: 状态正则化
from util_normalization import Normalization


set_seed(seed_number)

def multi_supervised_learning(relation, mode, extra_title ="",wandb_show=True,project_name="deep-path"):

    def train():
        # 从文件加载实体和关系的向量表示
        base_path = os.path.dirname(os.path.abspath(__file__))
        relation_path = os.path.join(dataPath, "tasks", relation, "train_pos")
        entity2vec_path = os.path.join(dataPath, "entity2vec.bern")
        relation2vec_path = os.path.join(dataPath, "relation2vec.bern")
        kb_env_rl_path = os.path.join(dataPath, "kb_env_rl.txt")
        # 保存模型和优化器的路径
        # save_model_path = os.path.join(
        #     base_path, 'torchmodels', f'supervised_{mode}_checkpoint_{relation}.pth')
        # save_optimizer_path = os.path.join(
        #     base_path, 'torchmodels', f'supervised_{mode}_checkpoint_{relation}_opt.pth')
        
        ppo_checkpoint_path = os.path.join(
            base_path, 'torchmodels', f'supervised_{mode}_checkpoint_{relation}.pth')

        
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
        # 创建初始的监督训练模型并保存
        model = PolicyNeuralNetTorch().to(device)
        # 使用Adam优化器，学习率为learning_rate
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,eps=custom_eps)

        # 从文件加载实体和关系的向量表示
        entity2vec = np.loadtxt(entity2vec_path)
        relation2vec = np.loadtxt(relation2vec_path)
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

        # trick： 状态正则化
        state_norm = None
        if use_state_norm:
            state_norm = Normalization(shape=(1,state_dim))



        # 对于每个训练样本
        for episode in range(num_samples):
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
                    sample[0], sample[1], 5, entity2vec, entity2id, relation2id, kb,log=False,state_norm=state_norm)
                # print("teacher生成好的强化学习训练数据数量", len(good_episodes))
            except Exception as e:
                print("Cannot find a path", e)
                continue

            loss = None
            # 对于每个好的训练数据
            for item in good_episodes:
                state_batch = []
                action_batch = []
                # 对于每一步的转换
                for t, transition in enumerate(item):
                    state_batch.append(transition.state)  # state: vector
                    action_batch.append(transition.action)  # action: ID
                state_batch = np.squeeze(state_batch)
                state_batch = np.reshape(
                    state_batch, [-1, state_dim])  # (3, 200)

                # 转换为PyTorch张量
                state_batch = torch.tensor(
                    state_batch, dtype=torch.float).to(device)  # (3, 200)
                action_batch = torch.tensor(action_batch).to(device)  # (3,)
                action_prob = model.forward(state_batch)  # (3, 400)
                # 这里的one_hot 是动作空间的类别映射
                onehot_action = one_hot(
                    action_batch, torch.tensor(action_space))  # (3, 400)
                action_mask = onehot_action.type(torch.bool)  # (3, 400)
                picked_action_prob = torch.masked_select(
                    action_prob, action_mask)  # (3,)
                # print(picked_action_prob.shape)
                # print(picked_action_prob)
                # input()
                loss = torch.sum(-torch.log(picked_action_prob))

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("loss",loss.item())
            # 可视化
            # print("loss",loss)    
            if wandb_show and loss is not None:
                wandb.log({"sl_loss": loss.item()})

        # 保存模型
        # torch.save(model, save_model_path)
        # torch.save(optimizer, save_optimizer_path)
        
        # 保存模型参数
        ppo_checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # 保存模型和优化器
        torch.save(ppo_checkpoint, ppo_checkpoint_path)

        
        print('model saved')
        if wandb_show:
            wandb.save(ppo_checkpoint_path)
            wandb.finish()
    # print("relationPath", relationPath)

    max_num_samples = 500  # 原始代码: 500

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    train()


if __name__ == "__main__":
    setproctitle("supervised")
    # 测试1个
    for relation in relations[:1]:
        print("relation", relation)
        multi_supervised_learning(relation,"PPO", wandb_show=False)

# 多进程
# if __name__ == "__main__":
#     torch.set_printoptions(threshold=10000)  
#     mode = "PPO"
#     wandb_show = True
#     extra_title = "_state_norm"
#     # 使用 ProcessPoolExecutor 开启多进程
#     with ProcessPoolExecutor(max_workers=min(len(relations),6)) as executor:
#         # 提交任务
#         futures = []
#         for relation in relations:
#             print("Relation:", relation)
#             future = executor.submit(multi_supervised_learning, relation, mode,extra_title, wandb_show)
#             futures.append(future)
        
#         # 等待所有任务完成
#         for future in futures:
#             future.result()
