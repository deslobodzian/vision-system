#!/bin/bash

nmcli con mod "Wired connection 1" \
        ipv4.addresses "10.56.87.20/24" \
        ipv4.gateway "10.56.87.1" \
        ipv4.method "manual"
ifconfig eth0
