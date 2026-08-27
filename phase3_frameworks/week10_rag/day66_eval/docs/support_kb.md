# Sensor Troubleshooting

## Sensor Offline

If your sensor shows as offline in the dashboard, first check the power LED.

A solid green LED means power is fine and the issue is connectivity.

A blinking red LED means the battery is below 15% and must be replaced.

No LED at all means the sensor has no power — check the wiring harness.

For connectivity issues with a solid green LED, the sensor may have lost its
LoRaWAN connection to the gateway.

Sensors reconnect automatically within 15
minutes.

If offline longer than 30 minutes, the gateway itself may be down.

## Gateway Down

A gateway is considered down if no sensors connected to it have reported in the
last 10 minutes. Check the gateway's ethernet connection and power. Gateways do
not run on battery — they require constant mains power. After restoring power,
allow 5 minutes for the gateway to re-establish connections with all sensors.

## Missed Alerts

If you did not receive an expected alert, first verify the alert was actually
triggered by checking the event log. An alert only sends notifications if the
detection confidence exceeds 80%. Detections below 80% are logged but do not
notify, to reduce false alarms. Check your notification settings to confirm
your contact method is verified.

# Account and Billing

## Subscription Tiers

The Basic tier covers up to 10 sensors and 1 gateway. The Pro tier covers up to
50 sensors and 5 gateways. Enterprise is unlimited. Exceeding your sensor limit
disables new sensor registration until you upgrade or remove sensors.

## Data Retention

Basic tier retains 30 days of detection history. Pro retains 1 year. Enterprise
retains indefinitely. Exported data is available as CSV regardless of tier.
