#!/bin/bash

VISION_SYSTEM_DIR="/opt/vision_system"
VISION_SYSTEM_EXECUTABLE="Vision_System"
VISION_SYSTEM_ENGINE_FILE="note.engine"
SOURCE_DIR="/path/to/your/source/files"

# Check if Vision_System is running
if systemctl --quiet is-active vision_system.service; then
  echo "VisionSystem service is already running. Stopping and disabling service."
  systemctl stop vision_system.service
  systemctl disable vision_system.service
  rm /lib/systemd/system/vision_system.service
  rm /etc/systemd/system/vision_system.service
  systemctl daemon-reload
  systemctl reset-failed
fi

# Make sure the VisionSystem directory exists
mkdir -p $VISION_SYSTEM_DIR

cp $SOURCE_DIR/$VISION_SYSTEM_EXECUTABLE $VISION_SYSTEM_DIR
cp $SOURCE_DIR/$VISION_SYSTEM_ENGINE_FILE $VISION_SYSTEM_DIR

cat > /lib/systemd/system/vision_system.service <<EOF
[Unit]
Description=Service for VisionSystem

[Service]
WorkingDirectory=$VISION_SYSTEM_DIR
ExecStart=/usr/bin/env $VISION_SYSTEM_DIR/$VISION_SYSTEM_EXECUTABLE
Type=simple
Restart=on-failure
RestartSec=5
Nice=-20
CPUSchedulingPolicy=rr
CPUSchedulingPriority=99
LimitRTPRIO=infinity
LimitRTTIME=infinity

[Install]
WantedBy=multi-user.target
EOF

cp /lib/systemd/system/vision_system.service /etc/systemd/system/
chmod 644 /etc/systemd/system/vision_system.service
systemctl daemon-reload
systemctl enable vision_system.service
systemctl start vision_system.service

echo "Created and started VisionSystem service."

