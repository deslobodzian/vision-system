#!/bin/bash

SOURCE_DIR="/home/orin/VisionSystem/build"
USER="orin"

VISION_SYSTEM_DIR="/opt/vision_system"
VISION_SYSTEM_EXECUTABLE="vision_system"
VISION_SYSTEM_ENGINE_FILE="note.engine"

ZMQ_TO_NETWORKTABLES_DIR="/opt/zmq_to_networktables"
ZMQ_TO_NETWORKTABLES_EXECUTABLE="zmq_to_networktables"

if systemctl --quiet is-active vision_system.service; then
    echo "VisionSystem service is already running. Stopping and disabling service."
    systemctl stop vision_system.service
    systemctl disable vision_system.service
    rm /lib/systemd/system/vision_system.service
    rm /etc/systemd/system/vision_system.service
    systemctl daemon-reload
    systemctl reset-failed
fi

if systemctl --quiet is-active zmq_to_networktables.service; then
    echo "zmq_to_networktables service is already running. Stopping and disabling service."
    systemctl stop zmq_to_networktables.service
    systemctl disable zmq_to_networktables.service
    rm /lib/systemd/system/zmq_to_networktables.service
    rm /etc/systemd/system/zmq_to_networktables.service
    systemctl daemon-reload
    systemctl reset-failed
fi

mkdir -p $VISION_SYSTEM_DIR
cp $SOURCE_DIR/$VISION_SYSTEM_EXECUTABLE $VISION_SYSTEM_DIR
cp $SOURCE_DIR/$VISION_SYSTEM_ENGINE_FILE $VISION_SYSTEM_DIR
chown -R $USER:$USER $VISION_SYSTEM_DIR

mkdir -p $ZMQ_TO_NETWORKTABLES_DIR
cp $SOURCE_DIR/zmq_to_networktables/$ZMQ_TO_NETWORKTABLES_EXECUTABLE $ZMQ_TO_NETWORKTABLES_DIR
chown -R $USER:$USER $ZMQ_TO_NETWORKTABLES_DIR

cat > /lib/systemd/system/vision_system.service <<EOF
[Unit]
Description=Service for VisionSystem

[Service]
WorkingDirectory=$VISION_SYSTEM_DIR
ExecStart=/usr/bin/taskset -c 0-5 /usr/bin/sudo -u $USER $VISION_SYSTEM_DIR/$VISION_SYSTEM_EXECUTABLE
Type=simple
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat > /lib/systemd/system/zmq_to_networktables.service <<EOF
[Unit]
Description=Service for zmq_to_networktables

[Service]
WorkingDirectory=$ZMQ_TO_NETWORKTABLES_DIR
ExecStart=/usr/bin/sudo -u $USER $ZMQ_TO_NETWORKTABLES_DIR/$ZMQ_TO_NETWORKTABLES_EXECUTABLE
Type=simple
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cp /lib/systemd/system/vision_system.service /etc/systemd/system/
chmod 644 /etc/systemd/system/vision_system.service
systemctl daemon-reload
systemctl enable vision_system.service
systemctl start vision_system.service

cp /lib/systemd/system/zmq_to_networktables.service /etc/systemd/system/
chmod 644 /etc/systemd/system/zmq_to_networktables.service
systemctl daemon-reload
systemctl enable zmq_to_networktables.service
systemctl start zmq_to_networktables.service

echo "Created and started VisionSystem and zmq_to_networktables services."

