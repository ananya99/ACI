import mujoco
import mujoco.viewer
import os

def main():
    # Create a simple environment with a floor and walls
    xml = """
    <mujoco>
        <include file="cambrian/models/agents/body.xml"/>
        <worldbody>
            <!-- Floor -->
            <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
            <!-- Walls -->
            <geom name="wall1" type="box" size="10 0.1 1" pos="0 10 1" rgba="0.5 0.5 0.5 1"/>
            <geom name="wall2" type="box" size="10 0.1 1" pos="0 -10 1" rgba="0.5 0.5 0.5 1"/>
            <geom name="wall3" type="box" size="0.1 10 1" pos="10 0 1" rgba="0.5 0.5 0.5 1"/>
            <geom name="wall4" type="box" size="0.1 10 1" pos="-10 0 1" rgba="0.5 0.5 0.5 1"/>
        </worldbody>
    </mujoco>
    """
    
    # Load the model
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Launch the viewer
    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        # Set camera position
        viewer.cam.distance = 15.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30

        # Keep the viewer open until user closes it
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Update the viewer
            viewer.sync()

if __name__ == "__main__":
    main() 