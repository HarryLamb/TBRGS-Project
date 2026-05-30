#Created by Ashton Foo

import math

A = -1.4648375
B = 93.75
SPEED_LIMIT_KMH = 60.0
INTERSECTION_DELAY_SECONDS = 30.0


def flow_to_speed(flow_vehicles_per_hour):
    flow = max(0.0, float(flow_vehicles_per_hour))

    discriminant = B ** 2 + 4 * A * (-flow)

    if discriminant < 0:
        return 32.0

    sqrt_d = math.sqrt(discriminant)
    speed_high = (-B - sqrt_d) / (2 * A)
    speed = min(SPEED_LIMIT_KMH, max(1.0, speed_high))

    return speed


def estimate_travel_time_seconds(distance_km, predicted_flow_15min):
    hourly_flow = float(predicted_flow_15min) * 4.0
    speed_kmh = flow_to_speed(hourly_flow)

    drive_time_seconds = (float(distance_km) / speed_kmh) * 3600.0
    total_time_seconds = drive_time_seconds + INTERSECTION_DELAY_SECONDS

    return total_time_seconds