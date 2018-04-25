# Tutorial
[TOC]
## 1 Environment
### 1.1 Overview
MaCa is a reinforcement learning aimed research platform for heterogeneous multi-agent cooperation and confrontation. The goal of MaCa is to provide a series of highly customized simulation task environments for multi-agent autonomous action decision and control technology development.

At present, MaCa integrates a customizable air combat simulation environment. This environment provides a 2D battlefield with two unit types, **detector** (AWACS) and **fighter**. Detectors have a L|S band long range omnidirectional radar. Fighters are equipped with an X band fire-control radar (FCR), a L|S|X band jammer and two types of missiles (long range and short range). The principal simulation features if the environment are as follows.

- Radar detection and electronic countermeasure (ECM) simulation. MaCa can simulate main lobe jamming and side lobe jamming in both barrage and spot mode.
- Multi-jammer based passive detection. If an unit's radar signal can be received by no less than 4 opposite side jammers at same time and this situation lasts at least 5 steps, the opposite side can location this unit by passive detection.
- Detection information sharing. The detection results of detectors and passive detection can be shared to all units on same side
- Probability based missile hitting accuracy algorithm. The hitting accuracy of a missile is related to missile type, target location source (FCR|detector|passive detection) and the number of observation point during the flight of the missile.

### 1.2 Simulation Rules

- Radar simulation:
  - The L band radar on detector has a omnidirectional detection view with max detection range 400 and 20 frequency points.
  - The S band radar on detector has a omnidirectional detection view with max detection range 400 and 20 frequency points.
  - The X band fire control radar has a 120° detection angle with detection max range 180 and 10 frequency points.
- Jammer simulation:
  - The L band jammer provides a 45° jamming sector with range 400.
  - The S band jammer provides a 30° jamming sector with range 400.
  - The X band jammer provides a 10° jamming sector with range 300.
  - Consistent with radar's frequency point, L and S band jammer has 20 frequency points, X band jammer has 10 frequency points.
  - Radar detection ranges under main lobe jamming are 120(L), 150(S), 30(X).
  - Radar detection ranges under side lobe spot jamming are 220(L), 260(S), 80(X).
  - Radar detection ranges under side lobe barrage jamming are 300(L), 320(S), 120(X).
  - Effect angles of main lobe jamming are 10°(L), 10°(S), 4°(X).
- Passive detection simulation: An enemy unit can be located if there are 4 jammers received it's radar signal at same time and this situation lasts at least 5 steps.
- Missile attack simulation:
  - Missile firing ranges are 120(long range missile) and 50(short range missile).
  - Missile hit probability algorithm is the following equation. ***P*** is final value of hit probability. ***Pb*** is missile's basic hit probability. ***Nmop*** is the number of missed observation points. ***Pr*** is the reduction probability each observation point lost, the value is 5%. *Ps* is the additional hit probability of target location source.

```math
P=(Pb-Nmop*Pr)*Ps
```

| Missile Type | Basic Hit Probability |
| - | - |
| Long range | 80% |
| Short range | 90% |

| Missile Type | Observation Point Requirement |
| - | - |
| Long range | 10 |
| Short range | 4 |

| Location Source | Additional Hit Probability |
| - | - |
| Detector L band radar | 20% |
| Detector S band radar | 30% |
| Fighter FCR | 90% |
| Passive detection | 60% |

### 1.3 Reinforcement Learning Interface
MaCa is a Multi-agent research platform, each agent has individual observation, action and reward.
#### 1.3.1 Observation
MaCa provides two types of observation structure. The image based observations are visualizing informations which are structured in a set of muti-dimension numpy arrays, such observations are suitable for most deep reinforcement learning method. Raw data observations are designed for other AI method e.g. hard coded agent and reinforcement learning without deep neural networks.
##### 1.3.1.1 Image based Observation
Image based observation is composed of spacial view and non-spacial feature.

- Spatial view consists of several rectangular channels. Each channel is a miniature battlefield view and represents one type of information. Background pixel value of a channel is 255, the effective information location will be placed a 3×3 square with other pixel value. Furthermore, if a unit is destroed, all the pixel will be set to -1 to indicate it's death state. All spatial observations in same type are organized into a four-dimensional array, so there are 3 spatial observation arrays for each side.
  - Detector spatial observation. Each detector's visualized observation, consists of 3 channels. The size of detector's spatial observation is **[detector_quantity × miniature_view_size_y × miniature_view_size_x × 3]**.
    1. This detector's position and radar status. Self position's pixel value will be set as radar status. 0: radar disabled, other value: radar enabled and frequency point value.
    2. This detector's radar detected enemy positions, enemy positions' pixel value will be set as each enemy's ID.
    3. This detector's radar detected enemy positions, enemy positions' pixel value will be set as each enemy's type, 0 is detector and 1 is fighter.
  - Fighter spatial observation. Each fighter's visualized observation consists of 9 channels. The size of fighter's spatial observation is **[fighter_quantity × miniature_view_size_y × miniature_view_size_x × 9]**.
    1. This fighter's position and radar status. Self position's pixel value will be set as radar status. 0: radar disabled, other value: radar enabled and frequency point value.
    2. This fighter's radar detected enemy positions, enemy positions' pixel value will be set as each enemy's ID.
    3. This fighter's radar detected enemy positions, enemy positions' pixel value will be set as each enemy's type, 0 is detector and 1 is fighter.
    4. This fighter's position and jammer status. Self position's pixel value will be set as jammer status. 0: jammer disabled, other value: jammer enabled and frequency point value.
    5. This fighter's jammer received enemy positions, enemy positions' pixel value will be set as each enemy's ID.
    6. This fighter's jammer received enemy positions, enemy positions' pixel value will be set as each enemy's type, 0 is detector and 1 is fighter.
    7. This fighter's jammer received enemy positions, enemy positions' pixel value will be set as each enemy's radar frequency point.
    8. Targets' positions that attacking by this fighter. Targets' positions' pixel value will be set as targets' IDs.
    9. Targets' positions that attacking by this fighter. Targets' positions' pixel value will be set as targets' types.
  - Joint spatial observation. Global visualized observation consists of 2 channels, all units can access. For now these observations indicate all enemies that detected by detectors and passive detection. The size of global joint spatial observation is **[1 × miniature_view_size_y × miniature_view_size_x × 3]**.
    1. All detectors' visible enemy positions and passive detection detected enemy positions, all positions' pixel value will be set as each enemy's ID.
    2. All detectors' visible enemy positions and passive detection detected enemy positions, all positions' pixel value will be set as each enemy's type.
    3. All detectors' visible enemy positions and passive detection detected enemy positions, all positions' pixel value will be set as each enemy's information source, 0: from detector, 1: from passive detection.
- Non-spatial feature records other informations that can not be expressed in image. All non-spatial observations in same type are organized into a two-dimensional array, so there are 2 non-spatial observation arrays for each side.
  - Detector non-spatial observation. Each detector's non-spatial observation is a 1 long linear array, so the size of detectors' observation is **[detector_quantity × 1]**.
    1. Course heading, the length is 1.
  - Fighter non-spatial observation. Each fighter's non-spatial observation is a 3 long linear array, so the size of detectors' observation is **[fighter_quantity × 3]**.
    1. Course heading, the length is 1.
    2. Remainder of long range missle, the length is 1.
    3. Remainder of short range missle, the length is 1.
    
![](https://leonfg.github.io/maca/resource/spatialobs-detector.png)
![](https://leonfg.github.io/maca/resource/spatialobs-fighter.png)
![](https://leonfg.github.io/maca/resource/spatialobs-joint.png)

##### 1.3.1.2 Raw data Observation
Raw data is another observation structure designed for some fixed-rule hard-code agents which are difficult to process image data. These observations are structured in list and dict mixed form.

**Detector observation**

```python
{'id':id, 'alive': alive status, 'pos_x': horizontal coordinate, 'pos_y': vertical coordinate, 'course': course, 'r_iswork': radar enable status, 'r_fre_point': radar frequency point, 'r_visible_list': radar visible enemy}
```

**Fighter observation**

```python
{'id':id, 'alive': alive status, 'pos_x': horizontal coordinate, 'pos_y': vertical coordinate, 'course': course, 'r_iswork': radar enable status, 'r_fre_point': radar frequency point, 'r_visible_list': radar visible enemy, 'j_iswork': jammer enable status, 'j_fre_point': jammer frequency point, 'j_recv_list': jammer received enemy, 'l_missile_left': long range missile left, 's_missile_left': short range missile left}
```

**Joint observation**

```python
{'strike_list': under attack enemy targets, 'passive_detection_enemy_list': enemies detected by passive detection}
```

#### 1.3.2 Action
##### 1.3.2.1 Detector Action
Each detector action structure is a 2 long linear array, and the detectors' action data is a two-dimensional array with size **[detector_quantity × 2]**.
- Course heading, the length is 1. Course is int value from 0 to 255, it indicates the direction angle.
- Radar control, the length is 1. This value range is [0, n], 0: radar shutdown, 1~n: radar is open and the frequency point n.

##### 1.3.2.2 Fighter Action
Each fighter action structure is a 5 long linear array, and the fighter' action data is a two-dimensional array with size **[fighter_quantity × 4]**.
- Course heading, the length is 1. Course is a int value from 0 to 255, it indicates the direction angle.
- Radar control, the length is 1. This value range is [0, n] (n = L:20, S:20, X:10), 0: radar shutdown, 1~n: radar is open and the frequency point n.
- Jammer control, the length is 1. This value range is [0, n+1] (n = L:20, S:20, X:10), 0: jammer shutdown, 1~n: jammer is open and the frequency point is n, n+1: barrage jammer.
- Attack control, the length is 1. This value range is [0, 2n] (n<=enemy unit quantity). 0: do not attack, 1~n: attack a target which ID is n with long range missile, n+1~2n: attack a target which ID is n with short range missile.

#### 1.3.3 Reward
There are several reward types during a combat.
- Attack validation reward. Because attack result (if a missile hit the target successfully) will come after several steps later, there will be no reward for an attack action after the same step. furthermore, tt needs to meet a lot of conditions to be able to launch a missile. MaCa provides an attack validation reward in order to give feedback for an attack action immediately and help the agent understand how it can launch a missile. If an agent's attack action is executable there will be a positive feedback to the agent, if not, there will be a negative feedback.
- Target hit reward. Positive feedback to an agent if it's missile hit a target successfully.
- Alive reward. If an unit is alive in a step, environment should give it a positive feedback to encourage agents to keep alive.
- Combat result reward. Win or lose feedback. There are several situations with different reward value.
  - Totally win
  - Totally lose
  - win
  - lose
  - draw

All above reward values can be customized in file [config.py](/config.py)

### 1.4 Environment API
The API of MaCa air combat environment is defined in [interface.py](/environment/interface.py). There are 4 types of APIs.

| API Set | Description |
| - | - |
| Environment | Environment initiation and control |
| LoadMap | Load a map and parse map information |
| PlayBack | Load a log and replay |
| Utilities | Some essential miscellaneous functions |
#### 1.4.1 Environment
Environment interface is a set of reinforcement learning style APIs which are defined in Class **Environment**.

| API Function | Description |
| - | - |
| __init__ | Environment instance initiation |
| reset | Reset everything to the original state for a new start |
| step | Run a simulation step |
| get_obs | Get image-based observation data, designed for some deep reinforcement learning methods |
| get_obs_raw | Get raw data observation |
| get_done | Check if the combat has finished |

**__init__**
```python
def __init__(self, size_x, size_y, side1_detector_list, side1_fighter_list, side2_detector_list, side2_fighter_list, max_step=5000,
             render=False, render_interval=1, random_pos=False, log=False, random_seed=-1):
    """
    Environment initiation
    :param size_x: battlefield horizontal size. got from LoadMap.get_map_size
    :param size_y: battlefield vertical size. got from LoadMap.get_map_size
    :param side1_detector_list: side 1 detector configuration. got from LoadMap.get_unit_property_list
    :param side1_fighter_list: side 1 fighter configuration. got from LoadMap.get_unit_property_list
    :param side2_detector_list: side 2 detector configuration. got from LoadMap.get_unit_property_list
    :param side2_fighter_list: side 2 fighter configuration. got from LoadMap.get_unit_property_list
    :param max_step: max step，0：unlimited
    :param render: display enable control, True: enable display, False: disable display
    :param render_interval: display interval, skip how many steps to display a frame
    :param random_pos: start location initial method. False: side 1 on right, side2 on left. True: random position on top, bottom, right and left)
    :param log: log control，False：disable log，other value：the folder name of log.
    :param random_seed: random digit，-1：generate a new one，other value：use an exist random digit value
    """
```

**reset**
```python
def reset(self):
    """
    Reset environment
    """
```

**step**
```python
def step(self, side1_detector_action, side1_fighter_action, side2_detector_action, side2_fighter_action):
    """
    Run a step. About action structure see 1.2.2
    :param side1_detector_action: Numpy ndarray [detector_quantity, 2]
    :param side1_fighter_action: Numpy ndarray [fighter_quantity, 4]
    :param side2_detector_action: Numpy ndarray [detector_quantity, 2]
    :param side2_fighter_action: Numpy ndarray [fighter_quantity, 4]
    :return: True, run succeed, False, run Failed
    """
```

**get_obs**
```python
def get_obs(self):
    """
    Get image-based observation. See 1.2.1.1
    :return: side1_detector_data. Numpy ndarray [detector_quantity, 1]
    :return: side1_fighter_data. Numpy ndarray [fighter_quantity, 3]
    :return: side1_detector_img. Numpy ndarray [detector_quantity, miniature_view_size_y, miniature_view_size_x, 3]
    :return: side1_fighter_img. Numpy ndarray [fighter_quantity, miniature_view_size_y, miniature_view_size_x, 9]
    :return: side1_joint_img. Numpy ndarray [1, miniature_view_size_y, miniature_view_size_x, 3]
    :return: side2_detector_data. Numpy ndarray [detector_quantity, 1]
    :return: side2_fighter_data. Numpy ndarray [fighter_quantity, 3]
    :return: side2_detector_img. Numpy ndarray [detector_quantity, miniature_view_size_y, miniature_view_size_x, 3]
    :return: side2_fighter_img. Numpy ndarray [fighter_quantity, miniature_view_size_y, miniature_view_size_x, 9]
    :return: side2_joint_img. Numpy ndarray [1, miniature_view_size_y, miniature_view_size_x, 3]
    """
```

**get_obs_raw**
```python
def get_obs_raw(self):
    """
    Get raw data observation. See 1.2.1.2
    :return: side1_detector_data
    :return: side1_fighter_data
    :return: side2_detector_data
    :return: side2_fighter_data
    detector obs:{'id':id, 'alive': alive status, 'pos_x': horizontal coordinate, 'pos_y': vertical coordinate, 'course': course, 'r_iswork': radar enable status, 'r_fre_point': radar frequency point, 'r_visible_list': radar visible enemy}
    fighter obs:{'id':id, 'alive': alive status, 'pos_x': horizontal coordinate, 'pos_y': vertical coordinate, 'course': course, 'r_iswork': radar enable status, 'r_fre_point': radar frequency point, 'r_visible_list': radar visible enemy, 'j_iswork': jammer enable status, 'j_fre_point': jammer frequency point, 'j_recv_list': jammer received enemy, 'l_missile_left': long range missile left, 's_missile_left': short range missile left}
    """
```

**get_done**
```python
def get_done(self):
    """
    Get done
    :return: done: True, False
    """
```

#### 1.4.2 Map
MaCa use map files to definition the battlefield size, unit quantities and unit properties. A class **LoadMap** is provided in [interface.py](/environment/interface.py) for map file loading and parsing.

| API Function | Description |
| - | - |
| __init__ | Map instance initiation |
| get_map_size | Return battlefield size |
| get_unit_num | Return quantities of each unit type |
| get_unit_property_list | Return unit propertie in list data type |

**__init__**
```python
def __init__(self, map_path):
    """
    Initial map class
    :param map_path: map path
    """
```

**get_map_size**
```python
def get_map_size(self):
    """
    Get map size
    :return: size_x: horizontal size
    :return: size_y: vertical size
    """
```

**get_unit_num**
```python
def get_unit_num(self):
    """
    Get unit quantity
    :return: side1_detector_num
    :return: side1_fighter_num
    :return: side2_detector_num
    :return: side2_fighter_num
    """
```

**get_unit_property_list**
```python
def get_unit_property_list(self):
    """
    Get unit config information
    :return: side1_detector_list, should be directly forward to Environment init interface
    :return: side1_fighter_list, should be directly forward to Environment init interface
    :return: side2_detector_list, should be directly forward to Environment init interface
    :return: side2_fighter_list, should be directly forward to Environment init interface
    """
```

#### 1.4.3 Replay
MaCa can save log in runtime for replay. Enable or disable log saving is controlled by **log** parameter in environment initiation (see 1.3.1.1). 

| API Function | Description |
| - | - |
| __init__ | Replay instance initiation |
| start | Replay begain |

**__init__**
```python
def __init__(self, log_name, display_delay_time=0):
    """
    Initial replay class
    :param log_name:
    :param display_delay_time:
    """
```

**start**
```python
def start(self):
    """
    Replay begin
    """
```

#### 1.4.4 Misc Utilities
| API Function | Description |
| - | - |
| get_distance | Return distance between two coordinates |
| angle_cal | Return direction angle from coordinate A to coordinate B |

```python
def get_distance(a_x, a_y, b_x, b_y):
    """
    Get distance between two coordinates
    :param a_x: point a horizontal coordinate
    :param a_y: point a vertical coordinate
    :param b_x: point b horizontal coordinate
    :param b_y: point b vertical coordinate
    :return: distance value
    """
```

```python
def angle_cal(o_x, o_y, e_x, e_y):
    """
    Get a direction (angle) from a point to another point.
    :param o_x: starting point horizontal coordinate
    :param o_y: starting point vertical coordinate
    :param e_x: end point horizontal coordinate
    :param e_y: end point vertical coordinate
    :return: angle value
    """
```

### 1.5 Map
MaCa uses .map files in folder [maps](/maps) to definition battlefield size, unit quantities and unit properties. In a map file, informations are stored in the form of JSON. The information elements are described as follows.

| IE | Data type | Description |
| - | - | - |
| map_name | string | Map name |
| size_x | int | Horizontal size of battlefield |
| size_y | int | vertical size of battlefield |
| side1_detector_num | int | Side 1 detector quantity |
| side1_fighter_num | int | Side 1 fighter quantity |
| side2_detector_num | int | Side 2 detector quantity |
| side2_fighter_num | int | Side 2 fighter quantity |
| side1_detector_list | array | Properties of side 1 detector |
| side1_fighter_list | array | Properties of side 1 fighter |
| side2_detector_list | array | Properties of side 2 detector |
| side2_fighter_list | array | Properties of side 2 fighter |

**sideX_detector_list:**

| IE | Data type | Description |
| - | - | - |
| speed | int | Move distance per step |
| r_band| int | Radar band. 0: L, 1: S |

**sideX_fighter_list**

| IE | Data type | Description |
| - | - | - |
| speed | int | Move distance per step |
| r_band| int | Radar band. 0: L, 1: S |
| j_band | int | Jammer band. 0: L, 1: S, 2: X |
| l_missile_num | int | Initial quantity of long-range missile |
| s_missile_num | int | Initial quantity of short-range missile |

[1000_1000_2_10_vs_2_10.map](/map/1000_1000_2_10_vs_2_10.map) is the default map of MaCa. It defines a 1000×1000 battlefield with 2 detectors and 10 fighters on each side. Unit properties are as follows.

**Detector：**

| Unit | Speed | Radar Band |
| - | - | - | - |
| Detector 1 | 1 | L |
| Detector 2 | 1 | S |

**Fighter:**

| Unit | Speed | Radar Band | Jammer Band | Long Range Missile | Short Range Missile |
| - | - | - | - | - | - |
| Fighter 1 | 2 | X | X | 2 | 4 |
| Fighter 2 | 2 | X | X | 2 | 4 |
| Fighter 3 | 2 | X | X | 2 | 4 |
| Fighter 4 | 2 | X | X | 2 | 4 |
| Fighter 5 | 2 | X | X | 2 | 4 |
| Fighter 6 | 2 | X | X | 2 | 4 |
| Fighter 7 | 2 | X | X | 2 | 4 |
| Fighter 8 | 2 | X | X | 2 | 4 |
| Fighter 9 | 2 | X | L | 2 | 4 |
| Fighter 10 | 2 | X | S | 2 | 4 |

### 1.6 Configuration and Customization
#### 1.6.1 Reward
Reward value is defined in class GlobalVar of file [reward.py](/configuration/reward.py), users can change the value to optimize training effect. The items in reward configuration are as follows.

| Item | Default Value | Description |
| - | :-: | - |
| reward_strike_detector_success | 6 | Missile hit a detector |
| reward_strike_detector_fail | 0 | Missile miss a detector |
| reward_strike_fighter_success | 5 | Missile hit a fighterr |
| reward_strike_fighter_fail | 0 | Missile miss a fighter |
| reward_detector_destroyed | -6 | A detector been destroyed |
| reward_fighter_destroyed | -5 | A fighter been destroyed |
| reward_strike_act_valid | 5 | A valid attack action |
| reward_strike_act_invalid | -5 | An invalid attack action |
| reward_keep_alive_step | 1 | Keep alive in a step |
| reward_totally_win | 200 | Round reward：totally win |
| reward_totally_lose | -200 | Round reward：totally lose |
| reward_win | 100 | Round reward：win |
| reward_lose | -100 | Round reward：lose |
| reward_draw | 0 | Round reward：draw |

Because of the latency of the missile hit result, environment provides  strike action valid reward as an immediate reward for training. This reward means if a fighter can launch a missile to attack a target, it depends on missile's range, distance between attacker and target and the visibility to attacker of the target.

#### 1.6.2 Environment
See [system.py](/configuration/system.py). At present, there is only one option item "img_obs_reduce_ratio" to indicate the reduction ratio of image based observation.

## 2 Agent Interface
Users can run a combat between two agents by [fight.py](/fight.py). In order to meet MaCa's calling requirements, a runnable agent should provides an entry **class Agent(BaseAgent)** in /agent/your-agent-name/agent.py. 

In class Agent, you must define the following methods.

**__init__:** Class initiation

```python
def __init__(self, size_x, size_y, detector_num, fighter_num):
    """
    Init this agent
    :param size_x: battlefield horizontal size
    :param size_y: battlefield vertical size
    :param detector_num: detector quantity of this side
    :param fighter_num: fighter quantity of this side
    """
```

**get_action:** Input observations and get actions. There two types of get_action, one is for image based observation and the other is for raw data observation. A agent must provides one of the two get action interfaces.

- For image based observation (structure of observation and action, see 1.2.1 and 1.2.2):

```python
def get_action(self, detector_data, fighter_data, detector_img, fighter_img, joint_img, step_cnt):
    """
    get actions
    :param detector_data: detectors' non-spatial observations. 
    :param fighter_data: fighters' non-spatial observations.
    :param detector_img: detectors' spatial observations.
    :param fighter_img: fighters' spatial observations.
    :param joint_img: global shared spatial observations.
    :param step_cnt: step count. int
    :return detector_action: detectors' actions.
    :return detector_action: fighters' actions.
    """
```

- For raw data observation

```python
def get_action(self, detector_obs_list, fighter_obs_list, joint_obs_dict, step_cnt):
    """
    get actions
    :param detector_obs_list:
    :param fighter_obs_list:
    :param joint_obs_dict:
    :param step_cnt: step count. int
    :return detector_action: detectors' actions.
    :return detector_action: fighters' actions.
    """
```

**get_api_type:** API type indication. This function is defined in the parent class [BaseAgent](/agent/base_agent.py). In the __init__ function of every agent class, the value "self.api_type" should be initiated to 0 or 1.

```python
def get_api_type(self):
    """
    get api type
    :return: API type. 0: Raw, 1: Image
    """
```

Please referencing [agent.py](/agent/fix_rule/agent.py), a fixed rule based agent.

## 3 Coding Procedure
There are several principal code block for running a combat:

**Step1 Load map, get map size, unit quantity and unit configuration**

```python
env_map = LoadMap(map_path)
size_x, size_y = env_map.get_map_size()
side1_detector_num, side1_fighter_num, side2_detector_num, side2_fighter_num = env_map.get_unit_num()
side1_detector_list, side1_fighter_list, side2_detector_list, side2_fighter_list = env_map.get_unit_property_list()
```

**Step2 Initiate agent**

```python
agent1_module = importlib.import_module(agent1_import_path)
agent2_module = importlib.import_module(agent2_import_path)
agent1 = agent1_module.Agent(size_x, size_y, side1_detector_num, side1_fighter_num)
agent2 = agent2_module.Agent(size_x, size_y, side2_detector_num, side2_fighter_num)
```

**Step3 Initiate environment**

```python
env = Environment(size_x, size_y, side1_detector_list, side1_fighter_list, side2_detector_list, side2_fighter_list)
```

**Step4 Get observation**

```python
side1_detector_data, side1_fighter_data, side1_detector_img, side1_fighter_img, side1_joint_img, side2_detector_data, side2_fighter_data, side2_detector_img, side2_fighter_img, side2_joint_img = env.get_obs()
```

**Step5 Get action**

```python
side1_detector_action, side1_fighter_action = agent1.get_action(side1_detector_data, side1_fighter_data, side1_detector_img, side1_fighter_img, side1_joint_img, step_cnt)
side2_detector_action, side2_fighter_action = agent2.get_action(side2_detector_data, side2_fighter_data, side2_detector_img, side2_fighter_img, side2_joint_img, step_cnt)
```

**Step6 Run a step**

```python
env.step(side1_detector_action, side1_fighter_action, side2_detector_action, side2_fighter_action)
```

**Step7 Get done**

```python
done = env.get_done()
```

**Step7 Check if it is done. If not done, repeat step 4-6. If done, finish**

## 4 Operation Example
### 4.1 Play a Combat with Trained Agents
[fight.py](/fight.py) can run a combat between two agents. Parameter definitions are as follows.

| Parameter | Necessity | Default value if inexistent | Description |
| - | - | - | - |
| --map | Optional | "1000_1000_2_10_vs_2_10" | Map name, only name, not file path |
| --agent1 | Optional | "fix_rule" | Agent 1 name, corresponding to the folder name under [/agent](/agent) |
| --agent2 | Optional | "fix_rule" | Agent 2 name, corresponding to the folder name under [/agent](/agent) |
| --round | Optional | 1 | How many rounds the combat will be played |
| --random_pos | Optional | N/A | If the initial positions are random or agent1 at left and agent2 at right |
| --log | Optional | N/A | If enable the log function |
| --log_path | Optional | "default_log" | log folder under [/log](/log). Taking effect only when --log exists |

The cmd to run a 10 rounds combat with map [1000_1000_2_10_vs_2_10.map](/maps/1000_1000_2_10_vs_2_10.map) between two instance of [fix_rule](/agent/fix_rule/agent.py) and recording log in folder "testlog" under [/log](/log) is as follow.

```bash
python fight.py --map=1000_1000_2_10_vs_2_10 --agent1=fix_rule --agent2=fix_rule --round=10 --log --log_path=testlog
```

### 4.2 Replay
[replay.py](/replay.py) can replay a specified log.

| Parameter | Necessity | Default value if inexistent | Description |
| - | - | - | - |
| log | Mandatory | N/A | log folder name under [/log](/log) |

The cmd to replay the log "default_log" is as follow.

```bash
python replay.py default_log
```

## 5 Train
