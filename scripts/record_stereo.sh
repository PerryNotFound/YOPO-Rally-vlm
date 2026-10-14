if [ -z "$1" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

FILENAME=$1
# Input file name
ros2 bag record -s mcap -o "$FILENAME" --storage-preset-profile zstd_fast \
  /oak/left/camera_info \
  /oak/left/image_raw \
  /oak/right/camera_info \
  /oak/right/image_raw