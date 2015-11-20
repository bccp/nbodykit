for i in `find examples -name 'run.sh'`; do
    bash $i || exit 1
done
