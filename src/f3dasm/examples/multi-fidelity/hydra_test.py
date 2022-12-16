from omegaconf import DictConfig, OmegaConf
import hydra
import os

@hydra.main(version_base=None, config_path=".", config_name="config")
def run_experiment(cfg: DictConfig):
    f = open(os.getcwd() + '/result.txt', "w")
    f.write('The value of a is ' + str(cfg.a))
    f.close()

if __name__ == "__main__":
    run_experiment()