#!/usr/bin/env bash
set -e
if [ ! -f upstreams.toml ]; then
  echo "Copy upstreams.toml.example to upstreams.toml and edit it."
  exit 1
fi
python -m src.proxy.server --config upstreams.toml
