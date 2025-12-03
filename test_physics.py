# test_physics.py
import pybullet as p
import pybullet_data
from env.cat_model import CatModel

cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81, physicsClientId=cid)

cat = CatModel()
body_id = cat.load()

print("physics client id:", cid)
print("body id:", body_id)
print("Num joints (getNumJoints):", p.getNumJoints(body_id, physicsClientId=cid))
for j in range(p.getNumJoints(body_id, physicsClientId=cid)):
    info = p.getJointInfo(body_id, j, physicsClientId=cid)
    # print joint index, name, joint type, parent frame pos
    print(f"Joint {j}: name={info[1]} type={info[2]} parentFramePos={info[14]}")

p.disconnect(cid)
