# view_maze_offscreen.py

import re
import mujoco
import numpy as np
import imageio

# 1) Read & patch the MJCF so schema passes
xml_path = "cambrian/models/mazes/maze_2_with_forest.xml"
with open(xml_path, 'r', encoding='utf-8') as f:
    xml = f.read()

# Rename <mujoco name="..."> → model="..."
xml = re.sub(
    r'<mujoco\s+name="([^"]+)"',
    r'<mujoco model="\1"',
    xml,
    count=1
)

# 2) Load from string (bypasses schema error on `name`)
model = mujoco.MjModel.from_xml_string(xml)
data  = mujoco.MjData(model)

# 3) Create an off‐screen GL context
width, height = 800, 800
ctx = mujoco.GLContext(width, height)
ctx.make_current()

# 4) Build scene, options & camera
scene = mujoco.MjvScene(model, maxgeom=model.ngeom)
opt   = mujoco.MjvOption()
cam   = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)

# Front‐view camera params
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat    = np.array([0.0, 0.0, 0.0])
cam.distance  = 10.0
cam.elevation = -30.0
cam.azimuth   = 45.0

# 5) Step once to populate scene
mujoco.mj_forward(model, data)

# 6) Render FRONT view → WebP
# Note: must pass a “catmask” bitmask between cam and scene
catmask = mujoco.mjtCatBit.mjCAT_ALL
mujoco.mjv_updateScene(model, data, opt, None, cam, catmask, scene)
mujoco.mjr_render(width, height, scene, ctx.ctx)
rgb, _ = ctx.read_pixels(width, height, depth=False)
rgb = np.flipud(rgb)
imageio.imwrite("maze_view_front.webp", rgb, format="WEBP")

# 7) Render TOP‐DOWN view
cam.elevation = -90.0
cam.distance  = 20.0
mujoco.mjv_updateScene(model, data, opt, None, cam, catmask, scene)
mujoco.mjr_render(width, height, scene, ctx.ctx)
rgb, _ = ctx.read_pixels(width, height, depth=False)
rgb = np.flipud(rgb)
imageio.imwrite("maze_view_top.webp", rgb, format="WEBP")

print("Saved: maze_view_front.webp & maze_view_top.webp")