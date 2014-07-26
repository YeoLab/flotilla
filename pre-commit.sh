#!/bin/bash

# Make sure code that isn't in the prospective commit doesn't get tested
git stash -q --keep-index

make lint && make test
RESULT=$?

# Get that other code we removed at the beginning, back
git stash pop -q

# Check if the result was empty
[ $RESULT -ne 0 ] && exit 1
exit 0

