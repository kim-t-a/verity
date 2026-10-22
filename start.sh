#!/usr/bin/env sh
# Start Verity and print the addresses to open.
#
# `chmod` is a no-op on the Windows-mounted checkout, so run this as:
#
#     bash start.sh            # start, reusing built images
#     bash start.sh --build    # rebuild after a code or dependency change
#
# Any extra arguments are passed through to `docker compose up`.
set -eu
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "No .env found. Copy .env.example to .env and fill in the provider values." >&2
    exit 1
fi

echo "Starting Verity..."
docker compose up -d --wait "$@"

# The WSL address is useful when the browser runs on another device; on
# Docker Desktop for Windows the 127.0.0.1 addresses are the ones to open.
host_ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)

echo
echo "  Verity is running"
echo
echo "  App        http://127.0.0.1:5173"
echo "  API docs   http://127.0.0.1:8000/docs"
echo "  Health     http://127.0.0.1:8000/api/health"
if [ -n "${host_ip:-}" ] && [ "$host_ip" != "127.0.0.1" ]; then
    echo "  Other device  http://$host_ip:5173  (same network; firewall may block)"
fi
echo
echo "  Logs   docker compose logs -f"
echo "  Stop   docker compose stop"
