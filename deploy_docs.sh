#!/usr/bin/env bash
BRANCH=gh-pages
TARGET_REPO=YeoLab/flotilla.git
PELICAN_OUTPUT_FOLDER=output

echo -e "Testing travis-encrypt"
echo -e "$VARNAME"

if [  ${TRAVIS_PYTHON_VERSION:0:1} == "2" ]; then
    echo -e "Starting to deploy to Github Pages\n"
    if [ "$TRAVIS" == "true" ]; then
        git config --global user.email "travis@travis-ci.org"
        git config --global user.name "Travis"
    fi
    #using token clone gh-pages branch
    git clone --quiet --branch=$BRANCH https://${GH_TOKEN}@github.com/$TARGET_REPO built_website > /dev/null

    git config credential.helper "store --file=.git/credentials"
    git remote set-url origin https://github.com/YeoLab/flotilla.git
    git config --global credential.helper "store --file=.git/credentials"
    git config --global user.email "olga.botvinnik@gmail.com"
    git config --global user.name "Olga Botvinnik"
    echo "https://${GH_TOKEN}:@github.com" > .git/credentials

    cd doc;
    make html;
    cd ../gh-pages;
    # add, commit and push files
    git add -f . ;
    git commit -m "Travis build $TRAVIS_BUILD_NUMBER pushed to Github Pages
    on $(date)"
    git commit -m "Update from Travis-CI on $(date)";
    git push -u origin gh-pages;

    rsync -rv --exclude=.git  ../$PELICAN_OUTPUT_FOLDER/* .



    git push -fq origin $BRANCH > /dev/null
    echo -e "Deploy completed\n"
fi