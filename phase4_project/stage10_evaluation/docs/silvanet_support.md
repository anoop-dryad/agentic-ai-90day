# Device Status States

Silvanet devices appear in one of several states in the Site Management App,
sorted into Active, Inactive, Fire Alert, and Calibrating tabs.

## Active
The device is reachable by the Silvanet Cloud and sending data normally. An
Active device is operating correctly.

## Inactive
The device is not reachable. A gateway is marked Inactive when it has missed
its last three scheduled messages to the Silvanet Cloud. Inactive status may be
caused by connectivity or charging issues, and needs to be resolved promptly to
ensure continued monitoring. If a Border Gateway becomes Inactive, all dependent
Mesh Gateways and Sensors will also appear inactive.

## Calibrating
A newly deployed device is in a 14-day calibration mode. During this period the
device is not yet providing reliable data or fire alerts. This is expected after
deployment and requires no action.

## Fire Alert
A Wildfire Sensor has transmitted a fire alert. This also appears as a blinking
icon in the app's title bar and requires immediate attention.

# Alerts, Warnings, and Notifications

Troubleshooting messages on a device are classified by severity:

- Alerts require immediate attention. The device is not functioning properly
  and the underlying issue must be resolved as soon as possible to ensure
  continued monitoring.
- Warnings indicate the device is not performing optimally. The issue should be
  resolved, but the device is still functioning normally.
- Notifications are information only and require no user action.

# Connectivity Troubleshooting

Silvanet Sensors connect to nearby Mesh or Border Gateways via LoRaWAN.
Connectivity depends on having at least one active gateway within range.

If a device loses connectivity, first check its energy status in the Site
Management App — connectivity can be lost if a device's energy level is
critically low. If the energy line in the device graph slopes downward and does
not recover, the connectivity loss is due to low energy. If the line stops while
energy remains, the device has likely lost contact with the Silvanet Cloud,
indicating a connectivity issue that may require physical inspection.

To improve sensor connectivity, ensure the device is deployed within range of a
gateway. Solutions include adding a Mesh Gateway or moving a Mesh Gateway closer
to out-of-range sensors. Each Mesh Gateway should connect to two or more other
gateways for redundancy.

# Charging and Energy

Silvanet devices are solar-powered and store energy in supercapacitors. If a
device has not charged or has not fully charged over the last 72 hours, a
charging warning is raised. Low energy is a common root cause of both Inactive
status and connectivity loss — always check energy before assuming a hardware
fault.

# Firmware Updates (FUOTA)

Silvanet supports Firmware Update Over The Air. All sensors in a Site are
updated together using multicast. The update can be stretched over up to a week
to accommodate low-power and regional requirements. Firmware transfers resume
safely after power interruptions.
