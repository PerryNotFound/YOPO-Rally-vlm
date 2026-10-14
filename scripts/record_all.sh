if [ -z "$1" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

FILENAME=$1
# Input file name
ros2 bag record -s mcap -o "$FILENAME" --storage-preset-profile zstd_fast \
  /camera/pose/sample \
  /end_state \
  /oak/left/image_rect/camera_info \
  /oak/left/image_rect/compressed \
  /oak/stereo/image_raw \
  /tf \
  /tf_static \
  /visualize/depth_nn \
  /visualize/goal_local \
  /visualize/odom_vel \
  /visualize/path_0 \
  /visualize/path_1 \
  /visualize/path_2 \
  /visualize/path_3 \
  /visualize/path_4 \
  /visualize/path_min_cost \
  /visualize/pos_end_0 \
  /visualize/pos_end_1 \
  /visualize/pos_end_2 \
  /visualize/pos_end_3 \
  /visualize/pos_end_4 \
  /visualize/pred_path \
  /visualize/ref_path