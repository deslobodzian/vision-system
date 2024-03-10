#!/bin/bash

nmcli con mod "Wired connection 1" \
        ipv4.method "auto" \
        ipv4.addresses "" \
        ipv4.gateway "" \
        ipv4.dns ""

systemctl restart NetworkManager

sleep 5

ifconfig eth0
