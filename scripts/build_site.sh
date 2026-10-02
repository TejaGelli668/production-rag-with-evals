#!/usr/bin/env bash
# Assemble the static project site into _site/ (served by GitHub Pages; also for local preview).
#   scripts/build_site.sh && python3 -m http.server -d _site 8080
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf _site
mkdir -p _site/img
cp -R site/. _site/
cp docs/img/*.svg docs/img/demo.gif _site/img/
touch _site/.nojekyll
echo "built _site/ ($(du -sh _site | cut -f1))"
