#!/bin/bash

# clear_cache.sh
# This script deletes common Python cache directories in the current project

# List of cache directories to remove
CACHE_DIRS=( "__pycache__" ".pytest_cache" ".mypy_cache" ".ruff_cache" ".tox" ".nox" )

echo "Clearing Python cache directories..."

# Loop through each cache directory and find/remove them
for dir in "${CACHE_DIRS[@]}"; do
    find . -type d -name "$dir" -exec rm -rf {} + 2>/dev/null
done

echo "Cache cleared successfully!"