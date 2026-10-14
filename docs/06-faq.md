# Frequently Asked Questions (FAQ)

## Simulation Related

**Q: Unity cannot connect after launching?**

Check whether `ros_tcp_endpoint` has launched properly (default listening on `0.0.0.0:10000`). Confirm that the IP address and port filled in Unity match the ROS2 machine, and that the firewall allows port 10000.

**Q: Cannot see candidate trajectories in RViz2?**

Ensure that the YOPO node acts properly after publishing a goal point (check if there are any messages on the `/end_state` topic), and that the corresponding Path displays (`/visualize/path_0` to `/visualize/path_4`) are added in RViz2.

**Q: The robot is not moving, and MPC outputs nothing?**

The MPC node requires both `/end_state` (from YOPO) and `/odom` to trigger planning. Check whether the odometry topic is being published normally and whether the YOPO node has received the goal point.

---

## Training Related

**Q: CloudCompare steps report errors when running `generate_data.py`?**

Confirm that CloudCompare 2.13+ is installed and its executable path is added to the system `PATH` (for the Flathub version, it is usually `flatpak run org.cloudcompare.CloudCompare`). Check if the invocation command in `cloud_compare.py` matches the local installation path.

**Q: Prompted that `data_opt.toml` cannot be found during training?**

You must run `generate_data.py` first to generate optimization labels before running the training script. Every Scene requires a corresponding `data_opt.toml`, otherwise that Scene will be skipped.

**Q: CasADi trajectory optimization fails for some frames?**

Some frames (e.g., when the starting point is near an obstacle) might fail to converge. These frames are automatically skipped and won't disrupt the overall training process. If the failure rate is too high, check if the velocity and sectoral angle constraints in `configs.py` match the simulator's motion range.

**Q: Slow training convergence or loss not decreasing?**

Confirm you are using PyTorch in the virtual environment (with CUDA support) by checking `torch.cuda.is_available()`. You can also try slightly reducing the learning rate or increasing the batch size.

---

## Real-World Deployment Related

**Q: T265 odometry drift causes unstable planning?**

T265 tends to accumulate errors over long distances. You can launch with `vins:=true` to switch to VINS-Fusion visual-inertial odometry, or integrate an IMU fusion solution onto the robot chassis.

**Q: OAK-D depth map resolution does not match training?**

The YOPO node will automatically resize the input depth map to 160×32, so there's no need to manually adjust the camera resolution. Make sure OAK-D is publishing the aligned depth map topic (`/oak/stereo/image_raw`).

**Q: MPC tracking is jittery / oscillating?**

You can slightly reduce `vel_out_multiplier` (e.g., down to 0.7~0.8) at the top of `mpc_node.py` or increase the control cost weight to suppress angular velocity oscillation.

**Q: Performance degrades after switching models?**

Confirm that the selected model matches the current code version (identical output channel counts). Different training experiments might use varying configurations for models; when switching, ensure to concurrently check parameters like `candidate_num` (default 5).