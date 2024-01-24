# python
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


# relation = sys.argv[1]
relation = "concept_agentbelongstoorganization"
if len(sys.argv) >= 2:
    relation = sys.argv[1]


base_path = os.path.dirname(os.path.abspath(__file__))
# relationPath = dataPath + "tasks/" + relation + "/" + "train_pos"
# dataPath = os.path.join(base_path, dataPath)
relationPath = os.path.join(dataPath, "tasks", relation, "train_pos")
save_model_path = os.path.join(
    base_path, 'torchmodels/policy_supervised_' + relation + ".pth")
save_optimizer_path = os.path.join(base_path, 'torchmodels/policy_supervised_' +
                                   relation + "_opt.pth")

print("relationPath", relationPath)

max_num_samples = 500  # 原始代码: 500

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def train():
    # 打开包含训练数据的文件
    f = open(relationPath)
    train_data = f.readlines()
    f.close()

    # 限制训练样本数量
    num_samples = min(len(train_data), max_num_samples)

    # 创建初始的监督训练模型并保存
    model = PolicyNeuralNetTorch().to(device)
    # 使用Adam优化器，学习率为learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)  # 删除weight_decay

    # 从文件加载实体和关系的向量表示
    entity2vec = np.loadtxt(dataPath + "entity2vec.bern")
    relation2vec = np.loadtxt(dataPath + "relation2vec.bern")
    # print(dataPath + "entity2vec.bern")
    # print("entity2vec.shape", entity2vec.shape)
    # print("relation2vec.shape", relation2vec.shape)
    entity2id, relation2id, relations = open_entity2id_and_relation2id()

    # 从文件加载知识图谱环境
    with open(dataPath + 'kb_env_rl.txt') as f:
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

        except Exception as e:
            print("Cannot find a path", e)
            continue

        # 对于每个好的训练数据
        for item in good_episodes:
            state_batch = []
            action_batch = []
            # 对于每一步的转换
            for t, transition in enumerate(item):
                state_batch.append(transition.state)  # state: vector
                action_batch.append(transition.action)  # action: ID
            state_batch = np.squeeze(state_batch)
            state_batch = np.reshape(state_batch, [-1, state_dim])  # (3, 200)

            # 转换为PyTorch张量
            state_batch = torch.tensor(
                state_batch, dtype=torch.float).to(device)  # (3, 200)
            action_batch = torch.tensor(action_batch).to(device)  # (3,)
            action_prob = model.forward(state_batch)  # (3, 400)
            onehot_action = one_hot(
                action_batch, torch.tensor(action_space))  # (3, 400)
            action_mask = onehot_action.type(torch.bool)  # (3, 400)
            picked_action_prob = torch.masked_select(
                action_prob, action_mask)  # (3,)
            loss = torch.sum(-torch.log(picked_action_prob))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 保存模型
    torch.save(model, save_model_path)
    torch.save(optimizer, save_optimizer_path)
    print('model saved')


if __name__ == "__main__":
    setproctitle("supervised")
    train()
