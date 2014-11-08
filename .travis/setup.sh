#!/bin/bash

# setup travis-ci configuration basing one the being-built branch

export DEPLOY_DOCS=false

if [[ $TRAVIS_BRANCH == 'master' ]] ; then
    export DEPLOY_HTML_DIR=docs-dev
    export DEPLOY_DOCS=true
elif [[ $TRAVIS_BRANCH =~ ^v[0-9.]+$ ]]; then
    export DEPLOY_HTML_DIR=docs
    export DEPLOY_DOCS=true
fi
