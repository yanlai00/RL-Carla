import carla
import math

def dist_to_roadline(carla_map, vehicle):
    curr_loc = vehicle.get_transform().location
    yaw = vehicle.get_transform().rotation.yaw
    waypoint = carla_map.get_waypoint(curr_loc)
    waypoint_loc = waypoint.transform.location
    waypoint_yaw = waypoint.transform.rotation.yaw
    lane_width = waypoint.lane_width
    yaw_diff = yaw - waypoint_yaw
    yaw_diff_rad = yaw_diff / 180 * math.pi

    bb = vehicle.bounding_box
    corners = bb.get_world_vertices(vehicle.get_transform())
    # print(len(corners))
    dis_to_left, dis_to_right = 100, 100
    for corner in corners:
        if corner.z < 1:
            waypt = carla_map.get_waypoint(corner)
            waypt_transform = waypt.transform
            waypoint_vec_x = waypt_transform.location.x - corner.x
            waypoint_vec_y = waypt_transform.location.y - corner.y
            dis_to_waypt = math.sqrt(waypoint_vec_x ** 2 + waypoint_vec_y ** 2)
            waypoint_vec_angle = math.atan2(waypoint_vec_y, waypoint_vec_x) * 180 / math.pi
            angle_diff = waypoint_vec_angle - waypt_transform.rotation.yaw
            if (angle_diff > 0 and angle_diff < 180) or (angle_diff > -360 and angle_diff < -180):
                dis_to_left = min(dis_to_left, waypoint.lane_width / 2 - dis_to_waypt)
                dis_to_right = min(dis_to_right, waypoint.lane_width / 2 + dis_to_waypt)
            else:
                dis_to_left = min(dis_to_left, waypoint.lane_width / 2 + dis_to_waypt)
                dis_to_right = min(dis_to_right, waypoint.lane_width / 2 - dis_to_waypt)

    return dis_to_left, dis_to_right, math.sin(yaw_diff_rad), math.cos(yaw_diff_rad)

def exist_intersection(carla_map, vehicle):
    curr_loc = vehicle.get_transform().location
    waypoint = carla_map.get_waypoint(curr_loc)
    waypoint_loc = waypoint.transform.location
    waypoints_to_end = waypoint.next_until_lane_end(100)
    waypoint_lane_end = waypoints_to_end[-1]
    dist_to_end = waypoint_loc.distance(waypoint_lane_end.transform.location)
    return dist_to_end