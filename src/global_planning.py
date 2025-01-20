import rospy
import tf2_ros
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from collections import deque

# Store colours matching UAS TW colour scheme as dict
COLOUR_SCHEME = {
    "darkblue": "#143049",
    "twblue": "#00649C",
    "lightblue": "#8DA3B3",
    "lightgrey": "#CBC0D5",
    "twgrey": "#72777A",
}


def deserialize_map(map_data):
    """Convert map data into free space and wall coordinates"""
    width = map_data.info.width
    height = map_data.info.height
    resolution = map_data.info.resolution
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y
    grid = np.array(map_data.data).reshape((height, width))

    free_space = []
    walls = []

    for y in range(height):
        for x in range(width):
            value = grid[y, x]
            world_x = x * resolution + origin_x
            world_y = y * resolution + origin_y

            if value == 0:
                free_space.append((world_x, world_y))
            elif value == 100:
                walls.append((world_x, world_y))

    return np.array(free_space), np.array(walls)


def world_to_map(x_world, y_world, resolution):
    """Convert world coordinates to map coordinates"""
    x_map = int((x_world) / resolution) * resolution
    y_map = int((y_world) / resolution) * resolution
    return x_map, y_map


def map_to_world(x_map, y_map, resolution):
    """Convert map coordinates to world coordinates"""
    x_world = x_map * resolution
    y_world = y_map * resolution
    return x_world, y_world


def get_map_from_ros():
    """Retrieve the map from ROS service"""
    rospy.wait_for_service("static_map")
    try:
        get_map_service = rospy.ServiceProxy("static_map", GetMap)
        response = get_map_service()
        return response.map
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return None


def get_robot_position():
    """Get the current robot position using tf2"""
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)
    try:
        transform = tf_buffer.lookup_transform(
            "map", "base_link", rospy.Time(), rospy.Duration(1.0)
        )
        return transform.transform.translation.x, transform.transform.translation.y
    except tf2_ros.TransformException as e:
        rospy.logerr(f"TF lookup failed: {e}")
        return None


def visualize_map(free_positions, wall_positions, robot_position):
    """Visualize the map with free space, walls, and robot position"""
    plt.rcParams["figure.figsize"] = [7, 7]
    fig, ax = plt.subplots()

    # Plot wall and free space positions
    ax.scatter(
        wall_positions[:, 0],
        wall_positions[:, 1],
        c=COLOUR_SCHEME["darkblue"],
        alpha=1.0,
        s=6**2,
        label="Walls",
    )
    ax.scatter(
        free_positions[:, 0],
        free_positions[:, 1],
        c=COLOUR_SCHEME["twgrey"],
        alpha=0.08,
        s=6**2,
        label="Unobstructed Space",
    )

    # Plot robot position
    if robot_position is not None:
        ax.scatter(
            robot_position[0],
            robot_position[1],
            c=COLOUR_SCHEME["twblue"],
            s=10**2,
            label="Robot Position",
        )

    # Configure the plot
    ax.set_xlabel("X-Coordinate [m]")
    ax.set_ylabel("Y-Coordinate [m]")
    ax.set_title("Map Data Transformed into World Coordinates")
    ax.grid(True)
    ax.legend()
    plt.show()


def create_graph(free_positions, wall_positions, map_data, step_size=1):
    """Create a graph representation of the map"""
    graph = {}
    discrete_free_positions = set(
        map(
            lambda pos: map_to_world(
                *world_to_map(pos[0], pos[1], step_size), step_size
            ),
            free_positions,
        )
    )
    wall_set = set(
        map(
            lambda pos: map_to_world(
                *world_to_map(pos[0], pos[1], map_data.info.resolution),
                map_data.info.resolution,
            ),
            wall_positions,
        )
    )

    for position in discrete_free_positions:
        x, y = position
        neighbors = [
            (x + step_size, y),
            (x - step_size, y),
            (x, y + step_size),
            (x, y - step_size),
        ]
        valid_neighbors = []
        for neighbor in neighbors:
            if neighbor in discrete_free_positions and not is_wall_between(
                position, neighbor, wall_set, map_data
            ):
                valid_neighbors.append((neighbor, step_size))
        graph[position] = valid_neighbors
    return graph


def is_wall_between(start, end, wall_set, map_data):
    """Check if there's a wall between two points"""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = ((dx**2) + (dy**2)) ** 0.5
    steps = int(distance / (map_data.info.resolution / 2))

    for i in range(1, steps):
        check_x = start[0] + dx * i / steps
        check_y = start[1] + dy * i / steps
        check_point = map_to_world(
            *world_to_map(check_x, check_y, map_data.info.resolution),
            map_data.info.resolution,
        )
        if check_point in wall_set:
            return True
    return False


def visualize_graph(wall_positions, robot_position, graph):
    """Visualize the graph representation of the map"""
    plt.rcParams["figure.figsize"] = [7, 7]
    fig, ax = plt.subplots()

    node_positions = np.array(list(graph.keys()))

    edge_lines = []
    for node, neighbors in graph.items():
        for neighbor, _ in neighbors:
            edge_lines.append([node, neighbor])
    edge_lines = np.array(edge_lines)

    # Plot walls and graph nodes
    ax.scatter(
        wall_positions[:, 0],
        wall_positions[:, 1],
        c=COLOUR_SCHEME["darkblue"],
        alpha=1.0,
        s=6**2,
        label="Walls",
    )
    ax.scatter(
        node_positions[:, 0],
        node_positions[:, 1],
        c=COLOUR_SCHEME["twblue"],
        alpha=1.0,
        s=8**2,
        label="Graph",
    )

    # Plot robot position if available
    if robot_position is not None:
        ax.scatter(
            [robot_position[0]],
            [robot_position[1]],
            c=COLOUR_SCHEME["twblue"],
            s=15**2,
            label="Robot Position",
        )

    # Plot graph edges
    for line in edge_lines:
        x0, y0 = line[0]
        x1, y1 = line[1]
        x = [x0, x1]
        y = [y0, y1]
        ax.plot(x, y, c=COLOUR_SCHEME["twblue"])

    # Configure the plot
    ax.set_xlabel("X-Coordinate [m]")
    ax.set_ylabel("Y-Coordinate [m]")
    ax.set_title("Graph Generated based on Map Data")

    ax.set_xticks([-1, 0, 1, 2, 3, 4])
    ax.set_yticks([-1, 0, 1, 2, 3, 4])

    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()
    plt.show()


def bfs(graph, start, goal):
    """Perform Breadth-First Search to find a path"""
    queue = deque([(start, [start])])
    visited = set([start])

    while queue:
        (node, path) = queue.popleft()
        if node == goal:
            return path

        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None  # No path found


def apply_bfs_to_graph(graph, start, goal):
    """Apply BFS to find a path in the graph"""
    if start not in graph or goal not in graph:
        return None

    path = bfs(graph, start, goal)
    return path


def visualize_graph_with_path(wall_positions, robot_position, graph, path):
    """Visualize the graph and the found path"""
    plt.rcParams["figure.figsize"] = [7, 7]
    fig, ax = plt.subplots()

    edge_lines = []
    for node, neighbors in graph.items():
        for neighbor, _ in neighbors:
            edge_lines.append([node, neighbor])
    edge_lines = np.array(edge_lines)

    # Plot walls
    ax.scatter(
        wall_positions[:, 0],
        wall_positions[:, 1],
        c=COLOUR_SCHEME["darkblue"],
        alpha=1.0,
        s=6**2,
        label="Walls",
    )

    # Plot robot position if available
    if robot_position is not None:
        ax.scatter(
            [robot_position[0]],
            [robot_position[1]],
            c=COLOUR_SCHEME["twblue"],
            s=15**2,
            label="Robot Position",
        )

    # Visualize the path with points
    if path:
        path_x, path_y = zip(*path)
        ax.plot(
            path_x, path_y, c=COLOUR_SCHEME["twblue"], linewidth=2, label="BFS Path"
        )
        ax.scatter(
            path_x,
            path_y,
            c=COLOUR_SCHEME["twblue"],
            s=8**2,
            zorder=3,
            label="Path Points",
        )

    # Configure the plot
    ax.set_xlabel("X-Coordinate [m]")
    ax.set_ylabel("Y-Coordinate [m]")
    ax.set_title("Graph and BFS Path")

    ax.set_xticks([-1, 0, 1, 2, 3, 4])
    ax.set_yticks([-1, 0, 1, 2, 3, 4])

    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()
    plt.show()


def publish_path_to_move_base(path):
    """Publish the path to move_base"""
    path_publisher = rospy.Publisher(
        "move_base_simple/goal", PoseStamped, queue_size=10
    )

    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "map"
    pose_msg.pose.orientation.x = 0.0
    pose_msg.pose.orientation.y = 0.0
    pose_msg.pose.orientation.z = 0.0
    pose_msg.pose.orientation.w = 1.0

    # Iterate over all nodes in the path
    for node in path:
        # Set the pose
        pose_msg.pose.position.x = float(node[0])
        pose_msg.pose.position.y = float(node[1])
        pose_msg.pose.position.z = 0.0
        # Publish the path
        rospy.loginfo(f"Published pose: x={node[0]}, y={node[1]}")
        path_publisher.publish(pose_msg)
        # Wait for the robot to reach the pose
        rospy.sleep(10)
    rospy.loginfo("Path published successfully")


if __name__ == "__main__":
    rospy.init_node("moro_maze_navigation")

    rospy.loginfo("Retrieving map from ROS...")
    map_data = get_map_from_ros()

    if map_data is not None:
        rospy.loginfo("Map retrieved successfully. Processing...")

        free_positions, wall_positions = deserialize_map(map_data)
        robot_position = get_robot_position()

        # MAP VISUALIZATION
        # rospy.loginfo("Visualizing the map...")
        # visualize_map(free_positions, wall_positions, robot_position)

        # GRAPH CREATION
        # rospy.loginfo("Creating graph...")
        # graph = create_graph(free_positions, wall_positions, map_data, step_size=1)

        # rospy.loginfo("Visualizing the graph...")
        # visualize_graph(wall_positions, robot_position, graph)

        # # PATH RECONSTRUCTION
        graph = create_graph(free_positions, wall_positions, map_data, step_size=1)

        # Define start and goal positions
        start_position = (2, 1)
        goal_position = (3, 3)

        rospy.loginfo("Applying BFS to find path...")
        path = apply_bfs_to_graph(graph, start_position, goal_position)

        # Visualize the graph and the path
        visualize_graph_with_path(wall_positions, robot_position, graph, path)

        if path:
            rospy.loginfo(f"Path found: {path}")
            # Publish the path to move_base
            rospy.loginfo("Publishing path to move_base...")
            publish_path_to_move_base(path)
        else:
            rospy.loginfo("No path found")
    else:
        rospy.logerr("Failed to retrieve the map. Exiting.")
