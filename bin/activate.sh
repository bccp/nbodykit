
if [[ -n $BASH_VERSION ]]; then
    _SCRIPT_LOCATION=${BASH_SOURCE[0]}
elif [[ -n $ZSH_VERSION ]]; then
    _SCRIPT_LOCATION=${funcstack[1]}
else
    echo "Only bash and zsh are supported"
    return 1
fi
DIRNAME=`dirname ${_SCRIPT_LOCATION}`

LOCAL=`cd $DIRNAME/../install;pwd`

for dir in `find $LOCAL -type d -name 'site-packages'`; do

    PYTHONPATH=$dir:$PYTHONPATH
done
export PYTHONPATH
