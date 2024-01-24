# python
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

# framework
from utils import *
from env import Env

# relation = sys.argv[1]
relation = "concept_agentbelongstoorganization"
if len(sys.argv) >= 2:
    relation = sys.argv[1]


# task = sys.argv[2]
task = "test"
# task = "retrain"

mode = "RL"
graphpath = dataPath + "tasks/" + relation + "/" + "graph.txt"
relationPath = dataPath + "tasks/" + relation + "/" + "train_pos"

max_episode_num = 300  # original code : 300
max_test_num = 500  # original code : 500

device = torch.device(
    cuda) if torch.cuda.is_available() else torch.device("cpu")


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
    onehot_action = one_hot(action.to(torch.int64), torch.tensor(action_space))
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
    train = training_pairs
    success = 0
    # path_found = set()
    path_found_entity = []
    path_relation_found = []
    entity2id, relation2id, relations = open_entity2id_and_relation2id()

    entity2vec = np.loadtxt(dataPath + "entity2vec.bern")
    relation2vec = np.loadtxt(dataPath + "relation2vec.bern")
    with open(dataPath + 'kb_env_rl.txt') as f:
        kb_all = f.readlines()
    kb = relation_to_kb(relation)
    eliminated_kb = []
    concept = "concept:" + relation
    for line in kb_all:
        rel = line.split()[2]
        if rel != concept and rel != concept + "_inv":
            eliminated_kb.append(line)

    for i_episode in range(num_episodes):
        start = time.time()
        print(f"Episode {i_episode}")
        print("Training sample: ", train[i_episode][:-1])

        env = Env(entity2vec, relation2vec, kb_all, eliminated_kb,
                  entity2id, relation2id, relations, train[i_episode])

        sample = train[i_episode].split()
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        episode = []
        state_batch_negative = []
        action_batch_negative = []
        for t in count():
            state_vec = idx_state(entity2vec, state_idx)
            # print(state_vec) TODO
            action_probs = predict(model, state_vec)
            # print(action_probs) TODO
            action_chosen = np.random.choice(
                np.arange(action_space), p=np.squeeze(action_probs))
            # 从整个elimited_kb中随机找到一个有效的新状态
            reward, new_state, done = env.interact(state_idx, action_chosen)

            if reward == -1:  # the action fails for this step
                state_batch_negative.append(state_vec)
                action_batch_negative.append(action_chosen)

            new_state_vec = idx_state(entity2vec, new_state)
            episode.append(Transition(
                state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

            if done or t == max_steps:
                break

            state_idx = new_state

        # Discourage the agent when it chooses an invalid step
        if len(state_batch_negative) != 0:
            print("Penalty to invalid steps:", len(state_batch_negative))
            # 将-0.05乘以损失
            update(model, np.reshape(state_batch_negative, (-1, state_dim)), -
                   0.05, action_batch_negative, optimizer)

        print("----- FINAL PATH -----")
        print("\t".join(env.path))
        print("PATH LENGTH", len(env.path))
        print("----- FINAL PATH -----")

        # If the agent successes, do one optimization
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

            # print("3up")
            update(model, np.reshape(state_batch, (-1, state_dim)),
                   total_reward, action_batch, optimizer)
        else:
            global_reward = -0.05
            # length_reward = 1/len(env.path)

            state_batch = []
            action_batch = []
            total_reward = global_reward
            for t, transition in enumerate(episode):
                if transition.reward == 0:
                    state_batch.append(transition.state)
                    action_batch.append(transition.action)
            # print("4up")
            update(model, np.reshape(state_batch, (-1, state_dim)),
                   total_reward, action_batch, optimizer)

            print("Failed, Do one teacher guidline")
            try:
                good_episodes = teacher(sample[0], sample[1], 1, env, kb)
                for item in good_episodes:
                    teacher_state_batch = []
                    teacher_action_batch = []
                    total_reward = 0.0*1 + 1*1/len(item)  # not used variable.
                    for t, transition in enumerate(item):
                        teacher_state_batch.append(transition.state)
                        teacher_action_batch.append(transition.action)
                    # print("5up")
                    update(model, np.squeeze(teacher_state_batch),
                           1, teacher_action_batch, optimizer)
                print("teacher guideline success")
            except Exception as e:
                print("Teacher guidline failed")
        print("Episode time: ", time.time() - start, "\n")
    print("Success percentage:", success/num_episodes)

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

    with open(get_path_stats_path(mode, relation), "w") as f:
        for item in relation_path_stats:
            f.write(item[0] + "\t" + str(item[1]) + "\n")
    print("Path stats saved")

    return


def retrain():
    print("Start retraining")
    with open(relationPath) as f:
        training_pairs = f.readlines()

    model = torch.load("torchmodels/policy_supervised_" + relation + ".pth")
    # implements L2 regularization by weight_decay (TODO needs verification)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # TODO deleted weight_decay, load optimizer
    optimizer = torch.load(
        'torchmodels/policy_supervised_' + relation + "_opt.pth")
    print("sl_policy restored")

    episodes = min(len(training_pairs), max_episode_num)
    REINFORCE(training_pairs, model, optimizer, episodes)
    torch.save(model, "torchmodels/policy_retrained_" + relation + ".pth")
    torch.save(optimizer, "torchmodels/policy_retrained_" +
               relation + "_opt.pth")
    print("Retrained model saved")


def test():
    entity2id, relation2id, relations = open_entity2id_and_relation2id()

    with open(log_save_path, "wt") as f:
        current_time = datetime.datetime.now()
        f.write("time:{current_time}\n")
    with open(relationPath) as f:
        test_data = f.readlines()
    test_num = min(len(test_data), max_test_num)

    success = 0

    path_found = list()
    path_relation_found = list()
    path_set = set()

    model = torch.load("torchmodels/policy_retrained_" + relation + ".pth")
    # implements L2 regularization by weight_decay (TODO needs verification)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # TODO deleted weight_decay, load optim
    optimizer = torch.load(
        'torchmodels/policy_retrained_' + relation + "_opt.pth")
    print("Model reloaded")

    entity2vec = np.loadtxt(dataPath + "entity2vec.bern")
    relation2vec = np.loadtxt(dataPath + "relation2vec.bern")
    with open(dataPath + 'kb_env_rl.txt') as f:
        kb_all = f.readlines()
    eliminated_kb = []
    concept = "concept:" + relation
    for line in kb_all:
        rel = line.split()[2]
        if rel != concept and rel != concept + "_inv":
            eliminated_kb.append(line)

    for episode in range(test_num):
        print(f"Test sample {episode}: {test_data[episode][:-1]}")
        with open(log_save_path, mode="at", encoding="utf-8") as f:
            f.write(f"\nTest sample {episode}: {test_data[episode][:-1]}\n")
        env = Env(entity2vec, relation2vec, kb_all, eliminated_kb,
                  entity2id, relation2id, relations, test_data[episode])
        sample = test_data[episode].split()
        # YY change
        # state_idx = [env.entity2id[sample[0]], env.entity2id[sample[1]], 0]
        state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]

        transitions = list()

        for t in count():
            # YY change
            # state_vec = env.idx_state(state_idx)

            state_vec = idx_state(env.entity2vec, state_idx)

            action_probs = predict(model, state_vec)

            action_probs = np.squeeze(action_probs)  # TODO 타입이 Tensor인데 될까?

            action_chosen = np.random.choice(
                np.arange(action_space), p=action_probs)
            reward, new_state, done = env.interact(state_idx, action_chosen)
            # YY change
            # new_state_vec = env.idx_state(new_state)
            new_state_vec = idx_state(env.entity2vec, new_state)

            transitions.append(Transition(
                state=state_vec, action=action_chosen, next_state=new_state_vec, reward=reward))

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
                # total_reward = 0.1*global_reward + 0.8*length_reward + 0.1*diverse_reward
                state_batch = []
                action_batch = []
                for t, transition in enumerate(transitions):
                    if transition.reward == 0:
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)
                update(model, np.reshape(state_batch, (-1, state_dim)),
                       0.1*diverse_reward, action_batch, optimizer)
            path_set.add(" -> ".join(env.path_relations))

    for path in path_found:
        rel_ent = path.split(" -> ")
        path_relation = []
        for idx, item in enumerate(rel_ent):
            if idx % 2 == 0:
                path_relation.append(item)
        path_relation_found.append(" -> ".join(path_relation))

    # path_stats = collections.Counter(path_found).items()
    relation_path_stats = list(
        collections.Counter(path_relation_found).items())
    relation_path_stats = sorted(
        relation_path_stats, key=lambda x: x[1], reverse=True)

    ranking_path = []
    for item in relation_path_stats:
        path = item[0]
        length = len(path.split(" -> "))
        ranking_path.append((path, length))

    ranking_path = sorted(ranking_path, key=lambda x: x[1])
    print("Success percentage:", success/test_num)
    with open(log_save_path, mode="at", encoding="utf-8") as f:
        f.write("Success percentage: " + str(success/test_num) + "\n")

    with open(dataPath + "tasks/" + relation + "/" + "path_to_use.txt", "w") as f:
        for item in ranking_path:
            f.write(item[0] + "\n")
    print("path to use saved")
    return


if __name__ == "__main__":
    torch.set_printoptions(threshold=10000)  # TODO for debug
    log_save_path = "policy_agent_testlog_" + relation + ".txt"
    print(device)
    if task == 'test':
        setproctitle("kimjw test")
        test()
    elif task == 'retrain':
        setproctitle("kimjw retrain")
        retrain()
    else:
        retrain()
        test()
