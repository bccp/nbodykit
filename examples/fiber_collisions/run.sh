DIR=`dirname $0`
echo Testing FiberCollisions
cd $DIR
[ -d ../output ] || mkdir ../output

python ../../bin/nbkit.py FiberCollisions test_fc.params