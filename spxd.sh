#!/bin/bash

# spxd bash script to install sphinx-deployment to your sphinx docs project.
#
############################################################################
# Usage:
#   spxd.sh [options]
#
# Options:
#   -h              Help
#   -p <docs_path>  Install sphinx_deployment to a specified docs path
############################################################################
# Example to install on <your_project/docs>
# $ cd <your_project>
# $ wget https://raw.github.com/teracy-official/sphinx-deployment/master/scripts/spxd.sh && chmod +x ./spxd.sh && ./spxd.sh -p ./docs
#

function command_exists() {
    type "$1" &> /dev/null;
}

function require() {
    if ! command_exists git ; then
        echo "Error: 'git' is required for installation, please install 'git' first."
        echo "Installation aborted!"
        exit 1
    fi
}

function usage() {
    echo "Usage:"
    echo "  spxd.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h             Help"
    echo "  -p <docs_path> Install sphinx_deployment to a specified docs path"
}

function install() {
    # assume that the current working directory is the root git repository directory
    # to copy travis-ci stuff into this directory
    local project_root_path=`pwd`
    # relative or absolute <docs_path>?
    if [[ $1 =~ ^\/ ]]; then
        local docs_path=$1
    else
        local docs_path="$project_root_path/$1"
    fi

    echo "installing sphinx_deployment to '$docs_path'..."
    cd /tmp
    rm -rf sphinx-deployment
    git clone https://github.com/teracy-official/sphinx-deployment.git
    cd sphinx-deployment
    git fetch origin
    git checkout origin/master
    # test
    # git clone https://github.com/hoatle/sphinx-deployment.git
    # cd sphinx-deployment
    # git fetch origin
    # git checkout origin/features/3_installation_bash_script
    # copy required stuff

    echo "copying required files..."
    mkdir -p $docs_path
    mkdir -p $docs_path/.deploy_heroku
    cp -r docs/* $docs_path
    cp docs/.gitignore $docs_path
    cp -r docs/.deploy_heroku/* $docs_path/.deploy_heroku
    cp .travis.yml $project_root_path
    mkdir -p $project_root_path/.travis
    cp -r .travis/* $project_root_path/.travis

    # copy meta stuff
    echo "copying meta files..."
    cp CHANGELOG.md $docs_path/CHANGELOG_sphinx_deployment.md
    cp LICENSE $docs_path/LICENSE_sphinx_deployment
    cp README.md $docs_path/README_sphinx_deployment.md

    # clean up
    cd ..
    rm -rf sphinx-deployment

    # add sphinx-deployment.mk to Makefile only if not added yet
    cd $docs_path
    if [ -f Makefile ] && ! grep -q sphinx_deployment.mk Makefile ; then
        echo '' >> Makefile
        echo 'include sphinx_deployment.mk' >> Makefile
    fi

    echo ''
    echo "installation completed, please read $docs_path/README_sphinx_deployment.md for usage."
}

# check requirements
require

while getopts ":p:h" opt; do
    case $opt in
        p)
            install $OPTARG
            exit 0
            ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid options -$OPTARG" >&2
            exit 1
            ;;
        :)
            if [ $OPTARG == "p" ]; then
                echo "Option -$OPTARG requires <docs_path> argument." >&2
            fi
            exit 1
            ;;
    esac
done
