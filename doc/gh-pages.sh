#!/bin/bash
cd ..
SOURCES="doc examples"
git checkout gh-pages
rm -rf _build _sources _static
git checkout master $SOURCES
cd doc/
make html
mv -fv _build/html/* ../
cd ..
rm -rf doc
touch .nojekyll
git add -A
git commit -m "generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`"
git push origin gh-pages
git checkout master
