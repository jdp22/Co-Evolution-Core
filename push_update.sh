#!/bin/bash
set -e
cd "$(dirname "$0")"
git add -A
git commit -m "Remove ES terminology; rename main file to island_finetuning.py"
git push
