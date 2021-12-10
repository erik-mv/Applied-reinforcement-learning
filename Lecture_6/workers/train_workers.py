from workers.env_workers import EnvPool
from trainers import build_trainer
from loggers.joint_logger import JointLogger
import time
from tqdm import tqdm


def train(config, env_name, device):
    trainer, policy, agent, exploration_agent = build_trainer(config["trainer"], device)
    env_pool = EnvPool(exploration_agent, env_name, config["wrapper_config"], config["n_workers"])
    logger = JointLogger(config["log_path"], config)

    env_steps_ctn = 0
    target_updates = config["target_updates"]
    steps_per_update = config["steps_per_update"]
    log_every = config["log_every"]
    update_policy_every = config["update_policy_every"]
    save_every = config["save_every"]

    initial_steps = config["initial_steps"]
    initial_transitions = env_pool.collect_experience(initial_steps)
    trainer.consume_transitions(initial_transitions, 0.)
    avg_update_time = 0.
    avg_experince_collection_time = 0.

    for updates in tqdm(range(target_updates)):
        updates += 1

        tmp_time = time.time()
        transitions = env_pool.collect_experience(steps_per_update)
        avg_experince_collection_time += time.time() - tmp_time

        tmp_time = time.time()
        log_info = trainer.consume_transitions(transitions, updates / target_updates)
        avg_update_time += time.time() - tmp_time

        env_steps_ctn += steps_per_update

        if updates % update_policy_every == 0:
            env_pool.update_agent(policy)

        if updates % log_every == 0:
            logger.set_step(env_steps_ctn)
            for k in log_info:
                logger.add_value(k, log_info[k])
            logger.add_value("avg_experience_collection_time", avg_experince_collection_time / updates)
            logger.add_value("avg_update_time", avg_update_time / updates)

        if updates % save_every == 0:
            logger.set_step(env_steps_ctn)
            logger.save_agent(policy)
