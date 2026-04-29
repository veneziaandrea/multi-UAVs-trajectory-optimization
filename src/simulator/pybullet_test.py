import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)      # per vedere qualcosa
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane = p.loadURDF("plane.urdf")
ball = p.loadURDF("sphere2.urdf",[0,0,2])

p.setGravity(0,0,-9.81)

for _ in range(1000):
    p.stepSimulation()
    time.sleep(1/240)