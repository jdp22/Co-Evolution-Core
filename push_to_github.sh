#!/bin/bash
set -e
cd "$(dirname "$0")"

git init
git add .
git commit -m "Initial commit: multi-island ES with marginal contribution reward"
git remote add origin https://github.com/jdp22/Co-Evolution-Core.git
git branch -M main
git push -u origin main
