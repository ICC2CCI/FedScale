"""Increase the advertised Flower heartbeat grace period on WAN links."""

from flwr.common import constant

# The default is 30s and the SuperLink patience is two intervals.  A 120s
# advertised interval tolerates a short Skupper reconnect without declaring a
# healthy, still-running SuperNode dead.  The resilient sender still emits
# heartbeats regularly using the same patched value.
constant.HEARTBEAT_DEFAULT_INTERVAL = 120
