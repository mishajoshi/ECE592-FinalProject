#!/usr/bin/env bash
set -euo pipefail

# Build attacker probes and synthetic victim
make -C attacker
make -C synthetic

echo "Build complete: attacker probes and synthetic victim."