#! /bin/bash


CHECKDIR=../checkpoints
CKPT=30000_3_100_m2.tar.gz

if [ ! -d $CHECKDIR ] ; then
    mkdir -p $CHECKDIR
fi

echo "Downloading checkpoints and unpacking to "
echo $CHECKDIR
cd $CHECKDIR

if [ -f $CKPT ] ; then 
    rm -f $CKPT
fi

wget -v http://cetus.usc.edu/files/$CKPT || exit 1
tar vzxf $CKPT || exit 1
rm -f $CKPT
cd $OLDPWD 

