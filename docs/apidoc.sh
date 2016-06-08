# bash

if ! python -c 'import numpydoc'; then easy_install --user numpydoc; fi
if ! python -c 'import sphinx'; then easy_install --user sphinx; fi

sphinx-apidoc -H "API Reference" -M -e -f -o api/ ../nbodykit ../nbodykit/plugins
exclude=__init__.py
for i in ../nbodykit/*/; do
    if [ $i == ../nbodykit/plugins/ ]; then continue; fi
    exclude="$exclude,$i"
done
for i in ../nbodykit/*.py; do
    exclude="$exclude,$i"
done
echo $exclude
sphinx-apidoc -H "Plugins Reference" -M -e -f -o plugins/ ../nbodykit $exclude
