#!/bin/sh
set -eu
case " $* " in
  *" --no-vcan "*) ;;
  *) ip link add dev vcanjarvis type vcan; ip link set vcanjarvis up ;;
esac
exec python3 service.py "$@"
