import torch
from torch import nn
from utils import *
import torch.optim as optim


# 增加ActorCritic网络


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fc1 = nn.Linear(state_dim, 512).to(device)
        nn.init.xavier_uniform_(fc1.weight)
        nn.init.constant_(fc1.bias, 0.0)

        fc2 = nn.Linear(512, 1024).to(device)
        nn.init.xavier_uniform_(fc2.weight)
        nn.init.constant_(fc2.bias, 0.0)

        fc3 = nn.Linear(1024, action_space).to(device)
        nn.init.xavier_uniform_(fc3.weight)
        nn.init.constant_(fc3.bias, 0.0)

        self.linear_relu_stack = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3,
            nn.Softmax(dim=1)
        )
        self.to(device)


    def forward(self, x):
        # 调出device输出加上to(device)
        device = x.device
        return self.linear_relu_stack(x).to(device)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fc1 = nn.Linear(state_dim, 256).to(device)
        nn.init.xavier_uniform_(fc1.weight)
        nn.init.constant_(fc1.bias, 0.0)
        fc2 = nn.Linear(256, 512).to(device) # 512， 1024
        nn.init.xavier_uniform_(fc2.weight)
        nn.init.constant_(fc2.bias, 0.0)
        fc3 = nn.Linear(512, 1).to(device) # 1024，1
        nn.init.xavier_uniform_(fc3.weight)
        nn.init.constant_(fc3.bias, 0.0)
        self.linear_relu_stack = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3,
            # nn.Softmax(dim=1)

            # nn.ReLU()  
        )
        self.to(device)

    def forward(self, x):
        # 调出device输出加上to(device)
        device = x.device
        return self.linear_relu_stack(x.float()).to(device).view(-1)
        # return self.linear_relu_stack(x).to(device).view(-1)
# 增加Actor-Critic Agent


# 预想的代理，但是原代码内的耦合度有点高，还是以网络的形式嵌入
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # 计算TD误差
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values

        # 计算Actor和Critic的损失
        self.optimizer_actor.zero_grad()
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(td_errors.detach() * log_probs).mean()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss = td_errors.pow(2).mean()
        critic_loss.backward()
        self.optimizer_critic.step()


# PPO
class ActorCriticNeuralNetTorch(nn.Module):

    def __init__(self):
        """
        包含两个部分，一个是Actor的网络结构，另一个是Critic的网络结构。
        在训练循环中，通过一次前向传播即可获取Actor和Critic的输出，
        然后分别计算Actor和Critic的损失，最后更新两者的参数。
        """
        super(ActorCriticNeuralNetTorch, self).__init__()
        # 定义Actor网络结构
        self.actor_fc1 = nn.Linear(state_dim, 512)
        self.actor_fc2 = nn.Linear(512, 1024)
        self.actor_fc3 = nn.Linear(1024, action_space)
        self.actor_linear_relu_stack = nn.Sequential(
            self.actor_fc1,
            nn.ReLU(),
            self.actor_fc2,
            nn.ReLU(),
            self.actor_fc3,
            nn.Softmax(dim=1)
        )

        # 定义Critic网络结构
        self.critic_fc1 = nn.Linear(state_dim, 512)
        self.critic_fc2 = nn.Linear(512, 1024)
        self.critic_fc3 = nn.Linear(1024, 1)
        self.critic_linear_relu_stack = nn.Sequential(
            self.critic_fc1,
            nn.ReLU(),
            self.critic_fc2,
            nn.ReLU(),
            self.critic_fc3
        )

    def forward(self, x):
        # 返回Actor和Critic的输出
        actor_output = self.actor_linear_relu_stack(x)
        critic_output = self.critic_linear_relu_stack(x)
        return actor_output, critic_output


class PolicyNeuralNetTorch(nn.Module):
    def __init__(self):
        super(PolicyNeuralNetTorch, self).__init__()
        fc1 = nn.Linear(state_dim, 512)
        nn.init.xavier_uniform_(fc1.weight)  # TODO 要不要添加 L2 正则化
        nn.init.constant_(fc1.bias, .0)
        fc2 = nn.Linear(512, 1024)
        nn.init.xavier_uniform_(fc2.weight)
        nn.init.constant_(fc2.bias, .0)
        fc3 = nn.Linear(1024, action_space)
        nn.init.xavier_uniform_(fc3.weight)
        nn.init.constant_(fc3.bias, .0)
        self.linear_relu_stack = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3,
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


# 略微下降
class PolicyNeuralNetTorchWithLN(nn.Module):
    def __init__(self):
        super(PolicyNeuralNetTorch, self).__init__()
        fc1 = nn.Linear(state_dim, 512)
        nn.init.xavier_uniform_(fc1.weight)
        nn.init.constant_(fc1.bias, .0)
        ln1 = nn.LayerNorm(512)  # 添加 Layer Norm
        fc2 = nn.Linear(512, 1024)
        nn.init.xavier_uniform_(fc2.weight)
        nn.init.constant_(fc2.bias, .0)
        ln2 = nn.LayerNorm(1024)  # 添加 Layer Norm
        fc3 = nn.Linear(1024, action_space)
        nn.init.xavier_uniform_(fc3.weight)
        nn.init.constant_(fc3.bias, .0)
        self.linear_relu_stack = nn.Sequential(
            fc1,
            ln1,  # 在激活函数之前添加 Layer Norm
            nn.ReLU(),
            fc2,
            ln2,  # 在激活函数之前添加 Layer Norm
            nn.ReLU(),
            fc3,
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


# 追加一个简单的自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        weighted_sum = torch.matmul(attn_probs, v)
        return weighted_sum

# 略微下降
class PolicyNeuralNetTorchWithAttention(nn.Module):
    def __init__(self):
        super(PolicyNeuralNetTorchWithAttention, self).__init__()
        self.state_dim = state_dim
        self.action_space = action_space

        fc1 = nn.Linear(state_dim, 512)
        nn.init.xavier_uniform_(fc1.weight)
        nn.init.constant_(fc1.bias, .0)
        # ln1 = nn.LayerNorm(512)
        
        # 添加自注意力机制
        self.self_attention = SelfAttention(512)
        
        fc2 = nn.Linear(512, 1024)
        nn.init.xavier_uniform_(fc2.weight)
        nn.init.constant_(fc2.bias, .0)
        # ln2 = nn.LayerNorm(1024)
        
        self.linear_relu_stack = nn.Sequential(
            fc1,
            # ln1,
            nn.ReLU(),
            self.self_attention,  # 使用自注意力机制
            fc2,
            # ln2,
            nn.ReLU(),
        )
        
        fc3 = nn.Linear(1024, action_space)
        nn.init.xavier_uniform_(fc3.weight)
        nn.init.constant_(fc3.bias, .0)
        self.softmax = nn.Softmax(dim=1)

        self.output_stack = nn.Sequential(
            fc3,
            self.softmax
        )

    def forward(self, x):
        features = self.linear_relu_stack(x)
        output = self.output_stack(features)
        return output




class PolicyNeuralNetTorchForBert(nn.Module):
    def __init__(self, applied_state_dim):
        super(PolicyNeuralNetTorchForBert, self).__init__()
        fc1 = nn.Linear(applied_state_dim, 512)
        # TODO L2 regularizer
        nn.init.xavier_uniform_(fc1.weight)
        nn.init.constant_(fc1.bias, .0)
        fc2 = nn.Linear(512, 1024)
        nn.init.xavier_uniform_(fc2.weight)
        nn.init.constant_(fc2.bias, .0)
        fc3 = nn.Linear(1024, action_space)
        nn.init.xavier_uniform_(fc3.weight)
        nn.init.constant_(fc3.bias, .0)
        self.linear_relu_stack = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3,
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class PolicyNeuralNetAttentionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        """constants"""
        trans_r_dim = 100
        model_dim = 768
        num_heads = 6
        """initialize <sos>"""
        self.sos = torch.zeros((1, 768))
        torch.nn.init.xavier_uniform_(self.sos)
        """layers"""
        self.trans_r_layer = nn.Linear(trans_r_dim, model_dim)
        self.attn_encoder = nn.MultiheadAttention(model_dim, num_heads)
        self.attn_decoder = nn.MultiheadAttention(model_dim, num_heads)
        """initialize layer parameters"""
        nn.init.xavier_uniform_(self.trans_r_layer.weight)
        nn.init.constant_(self.trans_r_layer.bias, .0)

    def forward(self, state):
        """expected shape
        state.size(): (model_dim * 2,)
        """
        trans_r_original = state[0]
        cls = state[1:]
        trans_r = self.trans_r_layer(trans_r_original)
        stack = torch.vstack((trans_r, *cls))
        attn1_res, _ = self.attn_encoder(stack, stack, stack)
        attn2_res, _ = self.attn_decoder(self.sos, attn1_res, attn1_res)
        return attn2_res



if __name__ == "__main__":
    # policy_nn(torch.rand(1, 200), 200, 3)
    print('networks.py')
