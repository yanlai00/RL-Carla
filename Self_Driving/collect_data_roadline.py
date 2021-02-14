import carla
import os
import random
import time
import logging
import argparse
import queue
import json
import numpy as np
import math
from utils import spawn_camera

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-t', '--tm-port',
        metavar='T',
        default=8000,
        type=int,
        help='Traffic Manager port to listen to (default: 8000)')
    argparser.add_argument(
        '-l', '--length',
        metavar='L',
        default=200,
        type=int,
        help='length of each episode (default: 200)')
    argparser.add_argument(
        '-e', '--nepisode',
        metavar='E',
        default=10,
        type=int,
        help='number of episodes (default: 10)')
    argparser.add_argument(
        '-s', '--save-path',
        metavar='S',
        default='sem_dis_roadline_depth/',
        type=str,
        help='folder to save the data. e.g. sem_dis_roadline_depth/')
    argparser.add_argument(
        '-m', '--map',
        metavar='M',
        default='Town02',
        type=str,
        help='map (default: Town02)')
    argparser.add_argument(
        '-r', '--resolution',
        metavar='R',
        default='48',
        type=str,
        help='resolution of semantic camera (default: 48)')
    argparser.add_argument(
        '--num-vehicles',
        default=0,
        type=int,
        help='number of vehicles')

    args = argparser.parse_args()
    save_path = args.save_path
    resolution = args.resolution
    num_vehicles = args.num_vehicles

    num_ego_vehicles = 1

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    try:
        iters = os.listdir(save_path + '/%s/' % args.map)
        iters.sort()
        start_iter = int(iters[-1]) + 1
    except:
        start_iter = 0

    print('Start collecting at', args.map)

    print('found %d episodes' % start_iter)

    for run_iter in range(args.nepisode):
        run_iter += start_iter
        print('start episode %d.' % run_iter)
        actor_list = []
        vehicle_list = []
        vehicles_list = []
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(30.0)
            world = client.load_world(args.map)
            settings = world.get_settings()
            settings.fixed_delta_seconds = 0.1
            settings.synchronous_mode = True     
            world.apply_settings(settings)
            blueprint_library = world.get_blueprint_library()

            traffic_manager = client.get_trafficmanager(args.tm_port)
            traffic_manager.set_hybrid_physics_mode(False)
            traffic_manager.global_percentage_speed_difference(0)

            # Create ego vehicle
            bp = random.choice(blueprint_library.filter("*lincoln*"))
            vehicles = []

            for _ in range(num_ego_vehicles):
                while True:
                    for attr in bp:
                        if attr.is_modifiable:
                            bp.set_attribute(attr.id, random.choice(attr.recommended_values))
                    bp.set_attribute('role_name', 'hero')
                    transform = random.choice(world.get_map().get_spawn_points())
                    vehicle = world.try_spawn_actor(bp, transform)
                    if not vehicle is None:
                        traffic_manager.ignore_lights_percentage(vehicle,100)
                        traffic_manager.distance_to_leading_vehicle(vehicle,0)
                        traffic_manager.vehicle_percentage_speed_difference(vehicle,-20)
                        traffic_manager.ignore_vehicles_percentage(vehicle, 20)
                        traffic_manager.ignore_walkers_percentage(vehicle, 20)
                        vehicles.append(vehicle)
                        break

            vehicle_list.extend(vehicles)

            # Create Sensors
            collision_hist = []
            lane_violation_hist = []
            for vehicle_id in range(num_ego_vehicles):
                collision_hist.append([])
                lane_violation_hist.append([])

            def _collision_data(event, vehicle_id):
                collision_actor_id = event.other_actor.type_id
                if collision_actor_id == 'static.road':
                    return
                collision_hist[vehicle_id].append(event)

            def _lane_violation_data(event, vehicle_id):
                lane_violation_hist[vehicle_id].append(event)

            colsensor_bp = world.get_blueprint_library().find('sensor.other.collision')
            lane_violation_sensor_bp = world.get_blueprint_library().find('sensor.other.lane_invasion')

            for vehicle_id in range(num_ego_vehicles):
                colsensor = world.spawn_actor(colsensor_bp, carla.Transform(), attach_to=vehicles[vehicle_id])
                lane_violation_sensor = world.spawn_actor(lane_violation_sensor_bp, carla.Transform(), attach_to=vehicles[vehicle_id])
                colsensor.listen(lambda event: _collision_data(event, vehicle_id))
                lane_violation_sensor.listen(lambda event: _lane_violation_data(event, vehicle_id))
                actor_list.append(colsensor)
                actor_list.append(lane_violation_sensor)


            # Currently using same type of ego car; if using different type of ego car, need to move this into for loop
            bound_x = vehicles[0].bounding_box.extent.x
            bound_y = vehicles[0].bounding_box.extent.y

            # Set camera transforms
            front_camera_transform = carla.Transform(carla.Location(x=bound_x, y= -bound_y, z=1.0), carla.Rotation(yaw=-90))
            back_camera_transform = carla.Transform(carla.Location(x=bound_x, y=bound_y, z=1.0), carla.Rotation(yaw=90))
            right_camera_transform = carla.Transform(carla.Location(x= -bound_x, y=bound_y, z=1.0), carla.Rotation(yaw=90))
            left_camera_transform = carla.Transform(carla.Location(x= -bound_x, y= -bound_y, z=1.0), carla.Rotation(yaw=-90))
            front_cameras = [None] * num_ego_vehicles * 2
            back_cameras = [None] * num_ego_vehicles * 2
            right_cameras = [None] * num_ego_vehicles * 2
            left_cameras = [None] * num_ego_vehicles * 2
            for vehicle_id in range(num_ego_vehicles):
                front_camera = spawn_camera(world, front_camera_transform, vehicles[vehicle_id], cam_types=['semantic'], image_size_x=resolution, image_size_y=resolution, fov="120")[0]
                back_camera = spawn_camera(world, back_camera_transform, vehicle, cam_types=['semantic'], image_size_x=resolution, image_size_y=resolution, fov="120")[0]
                right_camera = spawn_camera(world, right_camera_transform, vehicle, cam_types=['semantic'], image_size_x=resolution, image_size_y=resolution, fov="120")[0]
                left_camera = spawn_camera(world, left_camera_transform, vehicle, cam_types=['semantic'], image_size_x=resolution, image_size_y=resolution, fov="120")[0]

                front_camera_rgb = spawn_camera(world, front_camera_transform, vehicles[vehicle_id], cam_types=['depth'], image_size_x=resolution, image_size_y=resolution, fov="120")[0]
                back_camera_rgb = spawn_camera(world, back_camera_transform, vehicle, cam_types=['depth'], image_size_x=resolution, image_size_y=resolution, fov="120")[0]
                right_camera_rgb = spawn_camera(world, right_camera_transform, vehicle, cam_types=['depth'], image_size_x=resolution, image_size_y=resolution, fov="120")[0]
                left_camera_rgb = spawn_camera(world, left_camera_transform, vehicle, cam_types=['depth'], image_size_x=resolution, image_size_y=resolution, fov="120")[0]

                front_cameras[vehicle_id] = front_camera
                back_cameras[vehicle_id] = back_camera
                right_cameras[vehicle_id] = right_camera
                left_cameras[vehicle_id] = left_camera

                front_cameras[num_ego_vehicles + vehicle_id] = front_camera_rgb
                back_cameras[num_ego_vehicles + vehicle_id] = back_camera_rgb
                right_cameras[num_ego_vehicles + vehicle_id] = right_camera_rgb
                left_cameras[num_ego_vehicles + vehicle_id] = left_camera_rgb

                actor_list.extend([front_camera, back_camera, right_camera, left_camera])
                actor_list.extend([front_camera_rgb, back_camera_rgb, right_camera_rgb, left_camera_rgb])
            # End set camera transforms

            autopilot_states = [False] * num_ego_vehicles

            # Warm up the car with 20 frames, otherwise autopilot WON'T WORK!
            for _ in range(20):
                world.tick()
            
            # Warm up the car with another 20 frames
            for _ in range(20):
                world.tick()
                    
                throttle = random.choice([0.4, 0.5, 0.6, 0.7, 0.8])
                steer = random.choice([-0.5, 0, 0.5])
                brake = random.choice([0])
                vehicles[vehicle_id].apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            # End warmup

            front_image_queue = [None] * num_ego_vehicles * 2
            back_image_queue = [None] * num_ego_vehicles * 2
            right_image_queue = [None] * num_ego_vehicles * 2
            left_image_queue = [None] * num_ego_vehicles * 2

            for vehicle_id in range(num_ego_vehicles):
                front_image_queue[vehicle_id] = queue.Queue()
                front_cameras[vehicle_id].listen(front_image_queue[vehicle_id].put)
                back_image_queue[vehicle_id] = queue.Queue()
                back_cameras[vehicle_id].listen(back_image_queue[vehicle_id].put)
                right_image_queue[vehicle_id] = queue.Queue()
                right_cameras[vehicle_id].listen(right_image_queue[vehicle_id].put)
                left_image_queue[vehicle_id] = queue.Queue()
                left_cameras[vehicle_id].listen(left_image_queue[vehicle_id].put)

            for vehicle_id in range(num_ego_vehicles):
                front_image_queue[num_ego_vehicles + vehicle_id] = queue.Queue()
                front_cameras[num_ego_vehicles + vehicle_id].listen(front_image_queue[num_ego_vehicles + vehicle_id].put)
                back_image_queue[num_ego_vehicles + vehicle_id] = queue.Queue()
                back_cameras[num_ego_vehicles + vehicle_id].listen(back_image_queue[num_ego_vehicles + vehicle_id].put)
                right_image_queue[num_ego_vehicles + vehicle_id] = queue.Queue()
                right_cameras[num_ego_vehicles + vehicle_id].listen(right_image_queue[num_ego_vehicles + vehicle_id].put)
                left_image_queue[num_ego_vehicles + vehicle_id] = queue.Queue()
                left_cameras[num_ego_vehicles + vehicle_id].listen(left_image_queue[num_ego_vehicles + vehicle_id].put)

        
            for _ in range(args.length):
                
                stop_flag = False
                for vehicle_id in range(num_ego_vehicles):
                    if len(collision_hist[vehicle_id]) > 0 or len(lane_violation_hist[vehicle_id]) > 0:
                        stop_flag = True
                if stop_flag:
                    break

                world.tick()

                for vehicle_id in range(num_ego_vehicles):
                    vehicle = vehicles[vehicle_id]
                
                    throttle = random.uniform(0.25, 0.75)
                    steer = random.uniform(-0.5, 0.5)
                    brake = 0
                    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
                    time.sleep(0.1)

                    image = front_image_queue[vehicle_id].get()
                    image.save_to_disk(save_path + '/%s/%03d/agent%02d/%06d/front.png' % (args.map, run_iter, vehicle_id, image.frame))
                    back_image_queue[vehicle_id].get().save_to_disk(save_path + '/%s/%03d/agent%02d/%06d/back.png' % (args.map, run_iter, vehicle_id, image.frame))
                    right_image_queue[vehicle_id].get().save_to_disk(save_path + '/%s/%03d/agent%02d/%06d/right.png' % (args.map, run_iter, vehicle_id, image.frame))
                    left_image_queue[vehicle_id].get().save_to_disk(save_path + '/%s/%03d/agent%02d/%06d/left.png' % (args.map, run_iter, vehicle_id, image.frame))

                    front_image_queue[num_ego_vehicles + vehicle_id].get().save_to_disk(save_path + '/%s/%03d/agent%02d/%06d/front_rgb.png' % (args.map, run_iter, vehicle_id, image.frame))
                    back_image_queue[num_ego_vehicles + vehicle_id].get().save_to_disk(save_path + '/%s/%03d/agent%02d/%06d/back_rgb.png' % (args.map, run_iter, vehicle_id, image.frame))
                    right_image_queue[num_ego_vehicles + vehicle_id].get().save_to_disk(save_path + '/%s/%03d/agent%02d/%06d/right_rgb.png' % (args.map, run_iter, vehicle_id, image.frame))
                    left_image_queue[num_ego_vehicles + vehicle_id].get().save_to_disk(save_path + '/%s/%03d/agent%02d/%06d/left_rgb.png' % (args.map, run_iter, vehicle_id, image.frame))

                    location = vehicle.get_location()
                    velocity = vehicle.get_velocity()
                    angular_velocity = vehicle.get_angular_velocity()
                    acceleration = vehicle.get_acceleration()
                    control = vehicle.get_control()

                    waypoint = world.get_map().get_waypoint(location)
                    waypoint_loc = waypoint.transform.location
                    waypoint_yaw = waypoint.transform.rotation.yaw
                    lane_width = waypoint.lane_width
                    yaw = vehicle.get_transform().rotation.yaw
                    yaw_diff = yaw - waypoint_yaw
                    yaw_diff_rad = yaw_diff / 180 * math.pi

                    waypoints_to_end = waypoint.next_until_lane_end(100)
                    waypoint_lane_end = waypoints_to_end[-1]
                    dist_to_end = waypoint_loc.distance(waypoint_lane_end.transform.location)

                    bb = vehicle.bounding_box
                    corners = bb.get_world_vertices(vehicle.get_transform())

                    dis_to_left, dis_to_right = 100, 100
                    for corner in corners:
                        if corner.z < 1:
                            waypt = world.get_map().get_waypoint(corner)
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

                    aux_states = {}

                    aux_states['dis_to_left'] = dis_to_left
                    aux_states['dis_to_right'] = dis_to_right
                    aux_states['angle_diff_rad'] = yaw_diff_rad
                    aux_states['dis_to_end'] = dist_to_end


                    with open(save_path + '/%s/%03d/agent%02d/%06d/aux_states.json' % (args.map, run_iter, vehicle_id, image.frame), 'w') as f:
                        json.dump(aux_states, f)
                        f.close()

        # Set autopilot to False, destroy all actors and quit
        finally:
            for vehicle_id in range(num_ego_vehicles):
                autopilot_states[vehicle_id] = False
                vehicles[vehicle_id].set_autopilot(False, traffic_manager.get_port())
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
            for actor in actor_list:
                actor.destroy()
            for actor in vehicle_list:
                actor.destroy()
            print('done episode %d.' % run_iter)

if __name__ == '__main__':
    main()