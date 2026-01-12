# Planning and decision making

This is the code used for the course *47005 Planning and Decision Making*


## Installation

```sh
git clone https://github.com/ersinbo/planning-decision-making.git
cd planning-decision-making/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`
```

In order to run the simulations, go to the following folder:
```sh
cd project/
```
To run regular RRT,
```sh
python run_rrt.py
```
To run RRT_star,
```sh
python run_rrt_star.py
```
To run Kinodynamic RRT star,
```sh
python kino_rrt.py
```
