#! /bin/bash

if [[ -n "$TRAVIS_TAG" ]]; then
    if ! grep "$TRAVIS_TAG" "$1"; then
        echo Tag $TRAVIS_TAG does not match setup.py version. Bail.
        exit 1
    fi
fi
exit 0