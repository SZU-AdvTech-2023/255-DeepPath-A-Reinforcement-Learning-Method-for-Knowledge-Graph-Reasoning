
# framework
from env import Env
from utils import *
import os
import csv


from sl_policy_multi import multi_supervised_learning as sl_policy_multi_supervised_learning
from sl_policy_multi_actor_critic import multi_supervised_learning as sl_policy_multi_actor_critic_supervised_learning
from sl_policy_multi_ppo import multi_supervised_learning as sl_policy_multi_ppo_supervised_learning

from policy_agent_multi import policy_train_single as rl_retrain
from policy_agent_multi_actor_critic import policy_train_single as rl_retrain_actor_critic
from policy_agent_multi_PPO import policy_train_single as rl_retrain_ppo   
import concurrent.futures

# 在相同的参数下，一次性跑完全部三种方法，进行对比。


sl_methods = {
    "RL":sl_policy_multi_supervised_learning,
    "AC":sl_policy_multi_actor_critic_supervised_learning,
    "PPO":sl_policy_multi_ppo_supervised_learning
}

retrain_methods = {
    "RL":rl_retrain,
    "AC":rl_retrain_actor_critic,
    "PPO":rl_retrain_ppo
}

# 根据mode执行单个方法
def run_single_method(relation,mode,wandb_show,project_name):
    print(f"start run {mode} {relation}")
    rl_log_save_path = f"compare_log_{mode}_{relation}.txt"
    rl_log_save_path = os.path.join(os.path.dirname(__file__), "log", rl_log_save_path)
    sl_methods[mode](relation,mode=mode, wandb_show=wandb_show, project_name=project_name)
    success_ratio = retrain_methods[mode](relation, mode=mode, log_save_path=rl_log_save_path, wandb_show=wandb_show, project_name=project_name)
    return success_ratio




def main():
    project_name = "12-25-deeppath"
    for relation in relations:
        print("relation", relation)

        # 创建一个进程池
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 准备函数参数
            function_args = [
                (relation, "RL", True, project_name),
                (relation, "PPO", True, project_name),
            ]

            # 提交函数到进程池
            futures = [executor.submit(run_single_method, *args)
                       for args in function_args]
            # 实际上等价于
            # run_single_method(relation,"RL",wandb_show=True,project_name=project_name)
            # run_single_method(relation,"AC",wandb_show=True,project_name=project_name)
            # run_single_method(relation,"PPO",wandb_show=True,project_name=project_name)

            # 等待所有任务完成
            concurrent.futures.wait(futures)

            # 获取结果
            results = [future.result() for future in futures]

        csv_filename = "compare_result2.csv"
        csv_filename = os.path.join(os.path.dirname(
            __file__), "results", csv_filename)
        csv_header = ["relation", "RL", "PPO"]

        if not os.path.exists(csv_filename):
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(csv_header)

        # 将结果追加到CSV文件
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([relation] + results)


if __name__ == "__main__":
    main()