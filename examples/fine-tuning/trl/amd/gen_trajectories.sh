#!/usr/bin/env bash

while true; do
  # run your script
  python trajectory_gen_v2.py
  rc=$?

  # print a message after each run
  if [ $rc -eq 0 ]; then
    echo "✅ trajectory generation completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
  else
    echo "❌ trajectory generation exited with code $rc at $(date '+%Y-%m-%d %H:%M:%S')" >&2
  fi

  # wait 10 seconds before next run
  sleep 10
done
