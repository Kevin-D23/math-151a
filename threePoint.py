import numpy as np


def three_point_endpoint(x0, x1, x2, y0, y1, y2, h):
    return (1 / (2 * h)) * (-3 * y0 + 4 * y1 - y2)


def three_point_midpoint(x_prev, x0, x_next, y_prev, y0, y_next, h):
    return (1 / (2 * h)) * (y_next - y_prev)


# Given data
times = np.array([0, 3, 5, 8, 10, 13])
distances = np.array([0, 225, 383, 623, 742, 993])
actual_speeds = np.array(
    [75, 77, 80, 74, None, 72]
)  # Speed at t=10 is what we need to predict


def calculate_speeds():
    speeds = []

    # Calculate speed at each point using appropriate formula
    for i in range(len(times)):
        if i == 0:  # First point - use endpoint formula
            h = times[1] - times[0]
            speed = three_point_endpoint(
                times[0],
                times[1],
                times[2],
                distances[0],
                distances[1],
                distances[2],
                h,
            )
            speeds.append(speed)

        elif i == len(times) - 1:  # Last point - use endpoint formula backwards
            h = times[-1] - times[-2]
            speed = -three_point_endpoint(
                times[-1],
                times[-2],
                times[-3],
                distances[-1],
                distances[-2],
                distances[-3],
                h,
            )
            speeds.append(speed)

        else:  # Middle points - use midpoint formula
            h = (times[i + 1] - times[i - 1]) / 2
            speed = three_point_midpoint(
                times[i - 1],
                times[i],
                times[i + 1],
                distances[i - 1],
                distances[i],
                distances[i + 1],
                h,
            )
            speeds.append(speed)

    return speeds


if __name__ == "__main__":
    speeds = calculate_speeds()

    print("\nResults:")
    print("Time (s) | Distance (ft) | Calculated Speed (ft/s) | Actual Speed (ft/s)")
    print("-" * 65)

    for i in range(len(times)):
        actual = actual_speeds[i] if actual_speeds[i] is not None else "N/A"
        print(
            f"{times[i]:8.1f} | {distances[i]:12.1f} | {speeds[i]:19.2f} | {actual:^17}"
        )
