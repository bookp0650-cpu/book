# Retrieval Integration SAM3 Launch Guide

## Final operator procedure

Open the same interactive terminal used for the conventional integration, then
run:

```bash
cd /home/book/pro_book_SAM3/pro_hand_book_python
source /home/book/pro_book/pro_hand_book_python/.pro_hand_book_fixed/bin/activate
python3 Retrieval_integration_SAM3.py
```

No manual ROS setup command or SAM3 service command is required for this
interactive-terminal procedure.

To stop the integration, use the same termination procedure as the conventional
integration (normally Ctrl+C). If this invocation started SAM3, its recorded
service process is stopped during cleanup. A service that was already ready
before launch is not stopped.

## Why the conventional file could be run directly

This PC uses ROS 2 Humble, not ROS 1. An interactive Bash shell reads
`/home/book/.bashrc`, which:

- sources `/opt/ros/humble/setup.bash` at lines 118, 134, and 143;
- sets `ROS_DOMAIN_ID=20` at line 128;
- sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` at line 131;
- changes to `/home/book/pro_book/pro_hand_book_python` at line 135;
- activates `.pro_hand_book_fixed` at line 136; and
- sources the conventional colcon overlay
  `/home/book/pro_book/pro_hand_book_python/ros2_ws/install/setup.bash`
  at lines 145-146 when it exists.

The active values observed during investigation were:

```text
ROS_VERSION=2
ROS_DISTRO=humble
ROS_PYTHON_VERSION=3
ROS_DOMAIN_ID=20
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

ROS 2 uses DDS discovery and has no ROS 1 `roscore` requirement. Neither
`/home/book/.bashrc` nor `Retrieval_integration.py` starts `roscore`,
`roslaunch`, or a ROS 2 launch file.

`Retrieval_integration.py` initializes ROS itself with `rclpy.init()`, creates
the `book_retrieval_main` node, creates a `MultiThreadedExecutor`, and adds the
existing nodes to it. The integration uses the existing topics and callbacks,
including `/shelf_id`, `/navigation_goal`, `/navigation_goal_final`,
`/wall_distance`, `/target_mm`, `/input`, and `/move_tcp`. The SAM3 variant
does not add or alter any ROS topic, publisher, subscriber, or callback.

The integration does not launch external robot or navigation drivers. Those
nodes must already be available exactly as required by the conventional
operation. The xArm connection begins only when `main()` constructs `XArm7`;
`XArm7` constructs `XArmAPI` with the host from
`Retrieval_integration.yaml`. The SAM3 launch-management code runs before
`main()` and aborts before this robot connection if SAM3 cannot become ready.

## Environment locations

### Conventional integration environment

```text
/home/book/pro_book/pro_hand_book_python/.pro_hand_book_fixed
Python: /home/book/pro_book/pro_hand_book_python/.pro_hand_book_fixed/bin/python
Python version: 3.10.12
```

The environment contains the required `cv2`, NumPy, and `pyrealsense2`.
ROS `rclpy` is supplied by `/opt/ros/humble` after the ROS setup is sourced.
The new project directory supplies the `xarm7`, `linear_lift`,
`Dynamixel_win_pro_hand_book`, and `detection` modules through the current
working directory.

The venv activation script does not source ROS. ROS comes from `.bashrc` before
the venv is activated.

### ROS installation and workspace

```text
ROS setup: /opt/ros/humble/setup.bash
Automatically sourced workspace:
/home/book/pro_book/pro_hand_book_python/ros2_ws
Workspace setup:
/home/book/pro_book/pro_hand_book_python/ros2_ws/install/setup.bash
```

A copied workspace also exists at
`/home/book/pro_book_SAM3/pro_hand_book_python/ros2_ws`, but `.bashrc` currently
sources the conventional workspace shown above. `.bashrc` was intentionally
not changed.

### SAM3 service environment

```text
/home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam3_runtime/.venv
Python version: 3.12.13
torch: 2.10.0+cu128
```

The service listens on `127.0.0.1:8765` and reports readiness at:

```text
http://127.0.0.1:8765/health
```

Existing scripts:

```text
detection/pro_handbook/sam3_runtime/scripts/start_service.sh
detection/pro_handbook/sam3_runtime/scripts/stop_service.sh
```

The start script uses the SAM3 venv directly, records its PID in
`detection/pro_handbook/sam3_runtime/logs/service.pid`, and writes output to
`detection/pro_handbook/sam3_runtime/logs/service.stdout.log`. The service also
writes `detection/pro_handbook/sam3_runtime/logs/service.log`.

`start_service.sh` unsets `LD_LIBRARY_PATH` only in its own shell process and
the SAM3 child. The integration/ROS process keeps its CUDA and ROS library
paths unchanged.

### PaddleOCR environment

```text
/home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/.paadle_ocr
Python:
/home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/.paadle_ocr/bin/python
Python version: 3.10.12
Paddle: 3.2.2
PaddleOCR: 3.3.2
```

OCR remains an isolated subprocess and is not activated as the main
integration environment.

## Automatic SAM3 lifecycle

`Retrieval_integration_SAM3.py` uses
`detection.pro_handbook.sam3_runtime.integration_service_manager.Sam3ServiceSession`.
Importing that module supplies these defaults without changing `.bashrc`:

```text
BOOK_SEGMENTATION_BACKEND=sam3
SAM3_ENDPOINT=http://127.0.0.1:8765
PYTHONPATH includes /home/book/pro_book_SAM3/pro_hand_book_python
```

The manager performs the following before `main()`:

1. Request `/health`.
2. If it is already ready, use it and claim no ownership.
3. If the endpoint is reachable but still loading, wait without starting a
   competing service.
4. If the endpoint is unreachable, execute the existing `start_service.sh`,
   record its PID, and wait for `/health` to report `ready=true`.
5. Abort before `main()`, ROS initialization, or robot connection if readiness
   fails.

The ready timeout is 120 seconds. Measured model load plus warm-up on this PC
was approximately 15 seconds; 120 seconds provides an eight-times margin for a
cold filesystem and temporary GPU load while remaining bounded.

On normal completion, Python exceptions, and the existing SIGINT emergency
path, only the PID recorded by this invocation is passed to the existing stop
script. A changed PID file is treated as an ownership mismatch and is not
stopped.

## Conventional launch command

In the same interactive terminal environment:

```bash
cd /home/book/pro_book_SAM3/pro_hand_book_python
source /home/book/pro_book/pro_hand_book_python/.pro_hand_book_fixed/bin/activate
python3 Retrieval_integration.py
```

## Failure checks

Check readiness without starting robot code:

```bash
curl --fail-with-body http://127.0.0.1:8765/health
```

Check SAM3 logs:

```bash
tail -n 100 /home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam3_runtime/logs/service.stdout.log
tail -n 100 /home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam3_runtime/logs/service.log
```

Check the recorded PID:

```bash
cat /home/book/pro_book_SAM3/pro_hand_book_python/detection/pro_handbook/sam3_runtime/logs/service.pid
```

If launch reports that the endpoint is reachable but not ready, inspect the
health response and logs. Do not change the NVIDIA driver, system CUDA, ROS
setup, or either virtual environment.
