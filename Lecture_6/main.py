import json
from workers.train_workers import train


if __name__ == "__main__":
    env_name = "Breakout-v0"
    # При замене окружения или его параметров нужно также изменить:
    # 1. config["trainer"]["exploration_config"]["action_size"] на реальное количество доступных дискретных действий
    # 2. config["wrapper_config"], config["trainer"]["model"]["in_channels"] и config["trainer"]["model"]["in_features"]
    # 3. Архитектуру сети
    config_name = "apex.json"
    with open(f"configs/models/{config_name}", "r") as f:
        config = json.load(f)
    config["log_path"] = "logs/apex_test"  # Путь, по которому будут сохранены логи

    train(config, env_name, device="cuda")