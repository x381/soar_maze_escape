import rospy
import tf2_ros
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
import numpy as np
import copy
import time
from global_planning import get_map_from_ros, deserialize_map, create_graph, apply_bfs_to_graph

# Initialize ROS Node
try:
    rospy.init_node("local_navigation", anonymous=True)
except rospy.ROSException:
    pass

# Publishers for visualization and control
cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
path_pub = rospy.Publisher("/planned_path", Path, queue_size=10)
goal_pub = rospy.Publisher("/goal_pose", PoseStamped, queue_size=10)

# Global variables
map_data = None

# Callback for map data
def map_callback(msg):
    global map_data
    map_data = msg

# Subscriber to map
topic_map_sub = rospy.Subscriber("/map", OccupancyGrid, map_callback)

# Transform utilities
def pose2tf_mat(pose):
    """Convert pose (x, y, theta) to a transformation matrix"""
    x, y, theta = pose
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

def tf_mat2pose(tf_mat):
    """Convert transformation matrix back to (x, y, theta)"""
    x, y = tf_mat[0, 2], tf_mat[1, 2]
    theta = np.arctan2(tf_mat[1, 0], tf_mat[0, 0])
    return np.array([x, y, theta])

def localise_robot():
    """Retrieve the robot's current pose relative to the map frame"""
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    while True:
        try:
            trans = tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("Waiting for transform from 'map' to 'base_link'...")
            continue

    theta = R.from_quat([
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w
    ]).as_euler("xyz")[2]

    return np.array([
        trans.transform.translation.x,
        trans.transform.translation.y,
        theta
    ])

def is_collision_free(trajectory):
    """Check if the trajectory is collision-free using the map data"""
    global map_data
    if map_data is None:
        rospy.logwarn("Map data not available yet.")
        return True

    map_resolution = map_data.info.resolution
    map_origin = np.array([map_data.info.origin.position.x, map_data.info.origin.position.y])
    map_width = map_data.info.width
    map_height = map_data.info.height
    map_array = np.array(map_data.data).reshape(map_height, map_width)

    for pose in trajectory:
        x, y = pose[:2]
        map_x = int((x - map_origin[0]) / map_resolution)
        map_y = int((y - map_origin[1]) / map_resolution)

        if 0 <= map_x < map_width and 0 <= map_y < map_height:
            if map_array[map_y, map_x] > 50:  # Threshold for obstacle
                return False
        else:
            return False

    return True

def generate_controls(last_control):
    """Generate possible (v, w) control signals"""
    linear_range = np.linspace(-0.4, 0.4, 10)  # Faster linear velocity
    angular_range = np.linspace(-2.0, 2.0, 10)  # Faster angular velocity
    controls = np.array(
        [[v, w] for v in linear_range for w in angular_range]
    )
    return controls

def forward_kinematics(control, last_pose, dt):
    """Calculate the next pose given the control and current pose"""
    x, y, theta = last_pose
    vt, wt = control

    if wt == 0:
        dx = vt * dt
        dy = 0
    else:
        dx = -vt/wt * np.sin(theta) + vt/wt * np.sin(theta + wt * dt)
        dy = vt/wt * np.cos(theta) - vt/wt * np.cos(theta + wt * dt)

    dtheta = wt * dt

    return np.array([x + dx, y + dy, theta + dtheta])

def cost_fn(pose, goal_pose, control):
    """Calculate cost for the given pose, goal_pose, and control"""
    Q = np.diag([1.0, 1.0, 0.5])  # State weighting
    R = np.diag([0.1, 0.1])  # Control weighting

    error = np.abs(pose - goal_pose)
    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))  # Normalize theta

    cost = error.T @ Q @ error + control.T @ R @ control
    return cost

def evaluate_controls(controls, robot_pose, goal_pose, horizon, dt):
    """Simulate outcomes for each control and evaluate costs"""
    costs = []
    trajectories = []

    for control in controls:
        simulated_pose = robot_pose.copy()
        trajectory = []
        total_cost = 0

        for _ in range(horizon):
            simulated_pose = forward_kinematics(control, simulated_pose, dt)
            trajectory.append(simulated_pose)

        if is_collision_free(trajectory):
            for pose in trajectory:
                total_cost += cost_fn(pose, goal_pose, control)
        else:
            total_cost = float('inf')  # Discard trajectories leading to collisions

        costs.append(total_cost)
        trajectories.append(trajectory)

    return np.array(costs), trajectories

def publish_trajectory(trajectory):
    """Publish the trajectory as a Path message"""
    path_msg = Path()
    path_msg.header.frame_id = "map"
    path_msg.header.stamp = rospy.Time.now()

    for pose in trajectory:
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.orientation.z = np.sin(pose[2] / 2)
        pose_msg.pose.orientation.w = np.cos(pose[2] / 2)
        path_msg.poses.append(pose_msg)

    path_pub.publish(path_msg)

def main():
    rate = rospy.Rate(10)  # 10 Hz

    # Retrieve map data and generate BFS path
    map_data = get_map_from_ros()
    if map_data is None:
        rospy.logerr("Map data could not be retrieved.")
        return

    free_positions, wall_positions = deserialize_map(map_data)
    graph = create_graph(free_positions, wall_positions, map_data, step_size=1)
    start_position = (2, 1)  # Start position in map coordinates
    goal_position = (0, 0)  # Goal position in map coordinates
    bfs_path = apply_bfs_to_graph(graph, start_position, goal_position)

    # Convert BFS path to global path with orientation
    global_path = []
    for idx in range(len(bfs_path) - 1):
        x1, y1 = bfs_path[idx]
        x2, y2 = bfs_path[idx + 1]
        theta = np.arctan2(y2 - y1, x2 - x1)
    
        # Adjust x1 and y1 based on direction
        if x2 > x1:  # Moving right
            x1 += 0.5
        elif x2 < x1:  # Moving left
            x1 -= 0.5
    
        if y2 > y1:  # Moving up
            y1 += 0.5
        elif y2 < y1:  # Moving down
            y1 -= 0.5
    
        global_path.append([x1, y1, theta])

    # Add the last point with its orientation
    global_path.append([bfs_path[-1][0], bfs_path[-1][1], np.pi])  # Final goal orientation

    horizon = 10
    dt = 0.1
    last_control = np.array([0.0, 0.0])
    goal_index = 0

    stuck_timer_start = None
    stuck_timeout = 10  # 10 seconds
    last_position = None

    while not rospy.is_shutdown() and goal_index < len(global_path):
        goal_pose = global_path[goal_index]

        while not rospy.is_shutdown():
            robot_pose = localise_robot()
            controls = generate_controls(last_control)
            costs, trajectories = evaluate_controls(controls, robot_pose, goal_pose, horizon, dt)

            best_idx = np.argmin(costs)
            best_control = controls[best_idx]
            best_trajectory = trajectories[best_idx]

            # Publish control and trajectory
            cmd_msg = Twist()
            if costs[best_idx] == float('inf'):
                rospy.logwarn("No valid trajectory found. Skipping goal.")
                break

            cmd_msg.linear.x = best_control[0]
            cmd_msg.angular.z = best_control[1]
            cmd_vel_pub.publish(cmd_msg)

            publish_trajectory(best_trajectory)

            # Check if goal is reached
            error = np.linalg.norm(robot_pose[:2] - goal_pose[:2])
            if error < 0.1:
                rospy.loginfo(f"Goal {goal_index + 1} reached.")
                goal_index += 1
                stuck_timer_start = None
                last_position = None
                break

            # Stuck detection based on position change
            if last_position is None:
                last_position = robot_pose[:2]
                stuck_timer_start = time.time()
            else:
                position_change = np.linalg.norm(robot_pose[:2] - last_position)
                if position_change > 0.05:
                    last_position = robot_pose[:2]
                    stuck_timer_start = time.time()  # Reset timer if progress is made
                elif time.time() - stuck_timer_start > stuck_timeout:
                    rospy.logwarn("Robot appears to be stuck. Moving to the next goal.")
                    goal_index += 1
                    stuck_timer_start = None
                    last_position = None
                    break

            # Only warn if no progress after timeout
            if stuck_timer_start is not None and time.time() - stuck_timer_start > stuck_timeout:
                rospy.logwarn_once("Robot may be stuck but continuing attempts.")

            last_control = best_control
            rate.sleep()

    # Stop the robot
    cmd_msg = Twist()
    cmd_vel_pub.publish(cmd_msg)
    rospy.loginfo("All goals reached or exit achieved. Robot stopped.")

if __name__ == "__main__":
    main()
