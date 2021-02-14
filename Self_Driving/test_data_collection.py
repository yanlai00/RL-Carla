import carla
import os
import random

def main():
    vehicle_list = []
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(30.0)

        world = client.load_world('Town01')
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = True
        world.apply_settings(settings)
        blueprint_library = world.get_blueprint_library()

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_hybrid_physics_mode(True)

        # Create ego vehicle
        bp = random.choice(blueprint_library.filter("*lincoln*"))

        while True:
            transform = random.choice(world.get_map().get_spawn_points())
            vehicle = world.try_spawn_actor(bp, transform)
            if not vehicle is None:
                break

        vehicle_list.append(vehicle)
        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(vehicle,100)
        traffic_manager.distance_to_leading_vehicle(vehicle,0)
        traffic_manager.vehicle_percentage_speed_difference(vehicle,-20)
        traffic_manager.ignore_vehicles_percentage(vehicle, 20)
        traffic_manager.ignore_walkers_percentage(vehicle, 20)

        for _ in range(100):
            world.tick()

            vehicle.apply_control(carla.VehicleControl(throttle=1.0))

            location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            control = vehicle.get_control()

            print('location', location.x, location.y, location.z)
            print('velocity', velocity.x, velocity.y, velocity.z)
            print('control', control.throttle, control.brake, control.steer)

    # Set autopilot to False, destroy all actors and quit
    finally:
        for actor in vehicle_list:
            actor.destroy()

if __name__ == '__main__':
    main()