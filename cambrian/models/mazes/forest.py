import random
import xml.etree.ElementTree as ET
import numpy as np

def generate_perlin_noise_2d(width, height, scale=10.0, octaves=6, persistence=0.5):
    """
    Generates a 2D Perlin noise array.

    Args:
        width: Width of the output array.
        height: Height of the output array.
        scale: Scaling factor for the coordinates.  Larger values make the noise "tighter".
        octaves: Number of layers of noise to add. More octaves give more detail.
        persistence: How much each octave's amplitude is reduced.

    Returns:
        A 2D numpy array of floats, with values in the range [-1, 1].
    """
    def lerp(a, b, x):
        """Linear interpolation."""
        return a + x * (b - a)

    def fade(t):
        """Fade function for smooth interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(i, x, y):
        """Gradient vector calculation."""
        v = [
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
            [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
            [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
        ]
        h = i & 15
        u = x
        w = y
        if h < 4:
            return u, w
        elif h < 8:
            return y, x
        elif h < 12:
            return -u, w
        else:
            return y, -u

    def noise(x, y):
        """Calculate noise value at (x, y)."""
        X = int(x) & 255  # Ensure X is wrapped to 0-255 range
        Y = int(y) & 255  # Ensure Y is wrapped to 0-255 range
        x -= int(x)
        y -= int(y)
        fade_x = fade(x)
        fade_y = fade(y)
        
        # Wrap indices using modulo to prevent out-of-bounds access
        n00 = gradient(p[(X + p[Y]) & 255], x, y)
        n01 = gradient(p[(X + p[(Y + 1) & 255]) & 255], x, y - 1)  # Fix this line
        n10 = gradient(p[(X + 1 + p[Y]) & 255], x - 1, y)
        n11 = gradient(p[(X + 1 + p[(Y + 1) & 255]) & 255], x - 1, y - 1)  # Fix this line
        
        # Use lerp for smooth interpolation
        return lerp(lerp(n00[0] * x + n00[1] * y, n10[0] * (x - 1) + n10[1] * y, fade_x),
                    lerp(n01[0] * x + n01[1] * (y - 1), n11[0] * (x - 1) + n11[1] * (y - 1), fade_x),
                    fade_y)

    p = np.random.permutation(256)  # Randomize permutation table
    result = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            value = 0
            amplitude = 1.0
            frequency = 1.0
            for _ in range(octaves):
                value += noise(x / scale * frequency, y / scale * frequency) * amplitude
                amplitude *= persistence
                frequency *= 2.0
            result[y, x] = value
    return result

def generate_random_tree_positions(terrain_size, num_trees, min_distance):
    """
    Generates random tree positions on a terrain, ensuring a minimum distance
    between them.

    Args:
        terrain_size:  A tuple (width, depth) representing the size of the terrain.
        num_trees: The number of trees to generate.
        min_distance: The minimum distance allowed between tree positions.

    Returns:
        A list of (x, y) tuples representing the tree positions.
    """
    width, depth = terrain_size
    positions = []
    for _ in range(num_trees):
        for attempt in range(1000):  # Try a limited number of times
            x = random.uniform(-width / 2, width / 2)
            y = random.uniform(-depth / 2, depth / 2)
            valid = True
            for px, py in positions:
                if (x - px) ** 2 + (y - py) ** 2 < min_distance ** 2:
                    valid = False
                    break
            if valid:
                positions.append((x, y))
                break
        else:
            print(f"Warning: Could not place tree {len(positions) + 1} after 1000 attempts.")
            break  # Stop trying to place this tree
    return positions

def create_tree_xml(x, y, tree_variation):
    """
    Creates the XML elements for a tree at the given position.

    Args:
      x: The x-coordinate of the tree.
      y: The y-coordinate of the tree.
      tree_variation: Integer to select a variation.
    Returns:
      A list of ET.Element objects.
    """
    if tree_variation == 0:
        trunk_height = 2
        trunk_radius = 0.1
        crown_size = 0.3
    elif tree_variation == 1:
        trunk_height = 3
        trunk_radius = 0.15
        crown_size = 0.4
    elif tree_variation == 2:
        trunk_height = 1.6
        trunk_radius = 0.08
        crown_size = 0.25
    else:
        trunk_height = 2.4
        trunk_radius = 0.12
        crown_size = 0.35

    tree_body = ET.Element("body", {"name": f"tree_{x}_{y}", "pos": f"{x} {y} 0"})
    trunk_geom = ET.SubElement(tree_body, "geom", {
        "type": "cylinder",
        "size": f"{trunk_radius} {trunk_height / 2}",
        "fromto": f"0 0 0 0 0 {trunk_height}",
        "material": "bark"
    })

    # Create tree crown
    crown_geom = ET.SubElement(tree_body, "geom", {
        "type": "sphere",
        "size": f"{crown_size}",
        "pos": f"0 0 {trunk_height + crown_size}",
        "material": "leaves"
    })
    
    return tree_body

def generate_forest_xml(terrain_size, num_trees, min_distance, tree_variations):
    """
    Generates a MuJoCo XML string for a forest with random tree placements.

    Args:
      terrain_size: Tuple (width, depth) representing the terrain size.
      num_trees: The number of trees in the forest.
      min_distance: The minimum distance between trees.
      tree_variations: List of tree variations to choose from.
      
    Returns:
      A string representing the MuJoCo XML.
    """
    # Generate terrain heightmap using Perlin noise
    terrain_heightmap = generate_perlin_noise_2d(terrain_size[0], terrain_size[1])

    # Generate tree positions
    tree_positions = generate_random_tree_positions(terrain_size, num_trees, min_distance)
    
    # Create the XML root element
    worldbody = ET.Element("worldbody")

    # Add trees to the worldbody
    for x, y in tree_positions:
        tree_variation = random.choice(tree_variations)
        tree_xml = create_tree_xml(x, y, tree_variation)
        worldbody.append(tree_xml)

    # Create and return the final XML string
    model = ET.Element("mujoco", {"model": "forest"})
    model.append(worldbody)
    return ET.tostring(model, encoding="unicode", method="xml")

def save_xml_to_file(xml_element, file_name):
    """
    Saves the given XML element tree to a file.

    Args:
        xml_element: The root XML element (or the tree) to save.
        file_name: The path to the file where the XML should be saved.
    """
    tree = ET.ElementTree(xml_element)
    with open(file_name, "wb") as file:
        tree.write(file)

# Final XML structure
forest_xml = ET.Element("mujoco", {"model": "forest"})

# Add asset block with materials and optional textures
asset = ET.SubElement(forest_xml, "asset")
ET.SubElement(asset, "material", name="bark", rgba="0.55 0.27 0.07 1")
ET.SubElement(asset, "material", name="leaves", rgba="0.0 0.5 0.0 1")
ET.SubElement(asset, "material", name="grass", rgba="0.1 0.6 0.1 1")

# Optionally add a texture (if you want to use an image file later)
# ET.SubElement(asset, "texture", name="grass_tex", type="2d", file="grass.png")
# ET.SubElement(asset, "material", name="grass", texture="grass_tex", reflectance="0.2")

# Add worldbody with green ground
worldbody = ET.SubElement(forest_xml, "worldbody")

# Add green ground plane
ET.SubElement(worldbody, "geom", {
    "name": "ground",
    "type": "plane",
    "size": "10 10 0.1",  # Half of terrain size, adjust as needed
    "material": "grass",
    "rgba": "0.1 0.6 0.1 1",  # Green
    "pos": "0 0 0"
})

# Generate Perlin noise and tree positions
terrain_size = (20, 20)  # Example size
num_trees = 200  # Example number of trees
min_distance = 1  # Minimum distance between trees

# Generate terrain and tree positions
terrain = generate_perlin_noise_2d(terrain_size[0], terrain_size[1])
tree_positions = generate_random_tree_positions(terrain_size, num_trees, min_distance)

# Add trees to the XML structure
worldbody = ET.SubElement(forest_xml, "worldbody")
for position in tree_positions:
    x, y = position
    tree_variation = random.randint(0, 3)
    tree_body = create_tree_xml(x, y, tree_variation)
    worldbody.append(tree_body)

# Save the XML to a file
save_xml_to_file(forest_xml, "generated_forest.xml")
