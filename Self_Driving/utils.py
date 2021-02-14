import carla
import random
import time
import logging
import math
import numpy as np


def spawn_walkers(client, world, num_walkers, synchronous=True):
    '''
    Input: Client; World; Number of walkers to Spawn
    Output: List of Walker objects; List of controller and walker ids.
    '''
    SpawnActor = carla.command.SpawnActor
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')
    walkers_list = []
    spawn_points = []
    all_id = []
    for i in range(num_walkers):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    batch = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    if not synchronous:
        world.wait_for_tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # random max speed
        all_actors[i].set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)

    print('spawned %d walkers, press Ctrl+C to exit.' % (len(walkers_list)))

    return all_actors, all_id

def spawn_vehicles(client, world, num_vehicles):
    '''
    Input: Client; World; Number of vehicles to Spawn
    Output: List of Vehicle objects.
    '''
    vehicles_list = []
    batch = []
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for n, transform in enumerate(spawn_points):
        if n >= num_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

    for response in client.apply_batch_sync(batch):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    print('spawned %d vehicles, press Ctrl+C to exit.' % (len(vehicles_list)))
    return vehicles_list

def spawn_camera(world, transform, attach_to, cam_types='all', image_size_x="288", image_size_y="288", fov="100"):
    '''
    input: world, transform, attach_to, cam_types='all' (OR list of camera types)
    output: list of spawned cameras, in the order of 'rgb', 'depth', 'log_depth', 'semantic'
    example: 
    front_camera_transform = carla.Transform(carla.Location(x=2, z=1.4), carla.Rotation(0.0, 180.0, 0.0))
    front_rgb_camera, front_depth_camera, front_log_depth_camera, front_semantic_camera = spawn_camera(world, front_camera_transform, vehicle)
    '''
    if cam_types == 'all':
        cam_types = ['rgb', 'depth', 'log_depth', 'semantic']

    for cam_type in cam_types:
        assert cam_type in ['rgb', 'depth', 'log_depth', 'semantic'], 'unknown camera type'

    blueprint_library = world.get_blueprint_library()

    depth_camera_bp = blueprint_library.find('sensor.camera.depth')
    log_depth_camera_bp = blueprint_library.find('sensor.camera.depth')
    rgb_camera_bp = blueprint_library.find('sensor.camera.rgb')
    semantic_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    blueprints = [depth_camera_bp, log_depth_camera_bp, rgb_camera_bp, semantic_camera_bp]

    for blueprint in blueprints:
        blueprint.set_attribute("image_size_x", image_size_x)
        blueprint.set_attribute("image_size_y", image_size_y)
        blueprint.set_attribute("fov", fov)

    return_cameras = []

    if 'rgb' in cam_types :
        rgb_camera = world.spawn_actor(rgb_camera_bp, transform, attach_to=attach_to)
        return_cameras.append(rgb_camera)
    if 'depth' in cam_types:
        depth_camera = world.spawn_actor(depth_camera_bp, transform, attach_to=attach_to)
        return_cameras.append(depth_camera)
    if 'log_depth' in cam_types:
        log_depth_camera = world.spawn_actor(log_depth_camera_bp, transform, attach_to=attach_to)
        return_cameras.append(log_depth_camera)
    if 'semantic' in cam_types:
        semantic_camera = world.spawn_actor(semantic_camera_bp, transform, attach_to=attach_to)
        return_cameras.append(semantic_camera)

    assert len(return_cameras) == len(cam_types)

    return return_cameras
    


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = 35.0 * (math.sin(self._t) + 1.0)

    def __str__(self):
        return 'Sun(%.2f, %.2f)' % (self.azimuth, self.altitude)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.puddles = 0.0
        self.wind = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 75.0)
        self.wind = clamp(self._t - delay, 0.0, 80.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudyness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)


def get_active_light(ego_vehicle, world):

    _map = world.get_map()
    ego_vehicle_location = ego_vehicle.get_location()
    ego_vehicle_waypoint = _map.get_waypoint(ego_vehicle_location)

    lights_list = world.get_actors().filter("*traffic_light*")

    for traffic_light in lights_list:
        location = traffic_light.get_location()
        object_waypoint = _map.get_waypoint(location)

        if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
            continue
        if object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
            continue

        if not is_within_distance_ahead(
            location,
            ego_vehicle_location,
            ego_vehicle.get_transform().rotation.yaw,
            10.0,
            degree=60,
        ):
            continue

        return traffic_light

    return None

def is_within_distance_ahead(
    target_location, current_location, orientation, max_distance, degree=60
):
    u = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    distance = np.linalg.norm(u)

    if distance > max_distance:
        return False

    v = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])

    angle = math.degrees(math.acos(np.dot(u, v) / distance))

    return angle < degree

def distance(target_location, current_location):
    u = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    distance = np.linalg.norm(u)
    return distance