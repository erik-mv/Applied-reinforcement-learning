from agent.agent import Agent, ExplorationAgent
from .apex.trainer import ApexTrainer
from .apex.policy import ApexPolicy


def build_trainer(trainer_config, device):
    if trainer_config["name"] == "Apex":
        policy = ApexPolicy(trainer_config["model"])
        policy.to(device)
        agent = Agent(policy, device)
        exploration_agent = ExplorationAgent(policy, device, trainer_config["exploration_config"])
        trainer = ApexTrainer(policy, trainer_config["buffer_config"],
                              trainer_config.get("discount_factor", 0.99),
                              trainer_config.get("batch_size", 64),
                              trainer_config.get("updates_per_consumption", 16),
                              trainer_config.get("soft_update_tau", 0.01),
                              device)
    return trainer, policy, agent, exploration_agent