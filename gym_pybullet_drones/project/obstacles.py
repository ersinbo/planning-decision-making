import os
import random
import pkg_resources
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

class walls(CtrlAviary):
    def _addObstacles(self):
        """
        Add solid walls as obstacles, forming a rectangular enclosure.
        """
        base_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets')
        box_urdf = os.path.join(base_path, "box.urdf")

        wall_z = 0  # Height of the wall base

        # Fill the perimeter of x -1 to 1, y 1 to 2 with boxes, stacked 10 high
        import numpy as np
        step = 0.1  # Step size for filling the walls ~ 40 boxes per wall
        x_positions = np.arange(-1, 1 + step, step)
        y_positions = np.arange(1, 2 + step, step)

        for z_offset in range(10):  # 10 layers high
            z = wall_z + z_offset * 0.1  # Smaller spacing between layers
            # Bottom wall: y = 1, x from -1 to 1
            for x in x_positions:
                if not (z_offset >= 4 and z_offset <= 7 and -0.3 <= x <= 0.3):  # Skip middle for layers 4-6
                    pos = (x, 1, z)
                    if os.path.exists(box_urdf):
                        p.loadURDF(box_urdf,
                                   pos,
                                   p.getQuaternionFromEuler([0, 0, 0]),
                                   globalScaling=2.0,
                                   useFixedBase=True,
                                   physicsClientId=self.CLIENT)

            # Top wall: y = 2, x from -1 to 1
            for x in x_positions:
                if not (z_offset >= 4 and z_offset <= 7 and -0.3 <= x <= 0.3):  # Skip middle for layers 4-6
                    pos = (x, 2, z)
                    if os.path.exists(box_urdf):
                        p.loadURDF(box_urdf,
                                   pos,
                                   p.getQuaternionFromEuler([0, 0, 0]),
                                   globalScaling=2.0,
                                   useFixedBase=True,
                                   physicsClientId=self.CLIENT)

            # Left wall: x = -1, y from 1 to 2
            for y in y_positions:
                if not (z_offset >= 4 and z_offset <= 7 and 1.2 <= y <= 1.8):  # Skip middle for layers 4-6
                    pos = (-1, y, z)
                    if os.path.exists(box_urdf):
                        p.loadURDF(box_urdf,
                                   pos,
                                   p.getQuaternionFromEuler([0, 0, 0]),
                                   globalScaling=2.0,
                                   useFixedBase=True,
                                   physicsClientId=self.CLIENT)

            # Right wall: x = 1, y from 1 to 2
            for y in y_positions:
                if not (z_offset >= 4 and z_offset <= 7 and 1.2 <= y <= 1.8):  # Skip middle for layers 4-6
                    pos = (1, y, z)
                    if os.path.exists(box_urdf):
                        p.loadURDF(box_urdf,
                                   pos,
                                   p.getQuaternionFromEuler([0, 0, 0]),
                                   globalScaling=2.0,
                                   useFixedBase=True,
                                   physicsClientId=self.CLIENT)