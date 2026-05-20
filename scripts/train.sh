#!/usr/bin/env bash
set -e
export PYTHONIOENCODING=utf-8
case "${1:-smoke}" in
  smoke) modal run src/train/modal_train.py::smoke ;;
  full)  modal run src/train/modal_train.py::full ;;
  *) echo "usage: $0 [smoke|full]"; exit 1 ;;
esac
