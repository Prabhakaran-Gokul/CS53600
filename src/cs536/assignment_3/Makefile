# Makefile for Custom TCP Congestion Control Kernel Module

obj-m += tcp_custom.o

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(CURDIR)

all:
	$(MAKE) -C "$(KDIR)" M="$(PWD)" modules

clean:
	$(MAKE) -C "$(KDIR)" M="$(PWD)" clean

install: all
	sudo insmod tcp_custom.ko
	@echo "Module loaded. Verify with: sysctl net.ipv4.tcp_available_congestion_control"

remove:
	sudo rmmod tcp_custom
	@echo "Module removed."

test:
	@echo "Available congestion control algorithms:"
	@sysctl net.ipv4.tcp_available_congestion_control
	@echo ""
	@echo "Allowed congestion control algorithms:"
	@sysctl net.ipv4.tcp_allowed_congestion_control

set-custom:
	@echo "Adding custom to allowed list..."
	sudo sysctl -w net.ipv4.tcp_allowed_congestion_control="reno cubic custom"
	@echo "Setting custom as default..."
	sudo sysctl -w net.ipv4.tcp_congestion_control=custom
	@echo "Current default:"
	@sysctl net.ipv4.tcp_congestion_control

.PHONY: all clean install remove test set-custom
