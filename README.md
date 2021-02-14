# Reinforcement Learning and Data Collection for Self-Driving in Carla

Reinforcement Learning and Data Collection for self-driving in [Carla](https://github.com/carla-simulator/carla) with [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)

## Usage

### Reinforcement Learning

Go to the `Carla-RL` folder, and run:  
`
python train_sac.py --model-name sem_sac --width 96 --height 96 --repeat-action 4 --start-location highway --sensor semantic --episode-length 600
`
#### More training/evaluation options
`
python train_sac.py -h
`
#### Modify reward, goal condition, etc.
See `Carla-RL/carla_env.py`, and modify the relevant variable values in the `step()` function.

### Data Collection and Training Vision-based Predictors

`collect_data_roadline.py`: Data collection scripts. Saves image observations of the agent as well as other state information (e.g. distance to the center of the lane). The collect data can be used to train predictors offline. Sample Usage: `python collect_data_roadline.py`. Run `python collect_data_roadline.py -h` for more data collection configuration options.

`train_and_val_predictor_single.py`: training and validation of image-to-distance predictors using the collected dataset.

You can define pytorch custom datasets in the `datasets` folder and additional pytorch models in the `models` folder.  
`test_data_collection.py` provides a basic script which you can use to verify whether your data collection process works as expected. (e.g. whether the sensory observations are correctly incorporated into the dataset)

### Visualization of sensors
Modify `manual_control.py` to include the new sensors you added to the car, and (after you launched Carla) run
`
python manual_control.py
`
to see them in action.

## References
[The Official Carla Simulator Repo](https://github.com/carla-simulator/carla): includes basic tutorials and demos for controlling the car and get sensory infomation from the environment. 

[Carla-RL Repo from Sentdex](https://github.com/Sentdex/Carla-RL): includes a basic version of the gym-style reinforcement learning environment for Carla, and a basic DQN model.

[Oatomobile Research Framework for Self-Driving](https://github.com/OATML/oatomobile): A library for self-driving research with high-level APIs, baseline agents, and graphics setup.

