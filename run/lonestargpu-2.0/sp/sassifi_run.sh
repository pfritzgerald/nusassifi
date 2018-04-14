for i in ${APP_DIR}/../../inputs/random-16800-4000-3.cnf; do
    MAXCLAUSES=`echo $i | sed 's/.*-\([0-9]\+\).cnf/\1/'`
#    echo $i
    ${APP_DIR}/nsp.sh ${BIN_DIR}/nsp $i $MAXCLAUSES >> stdout.txt 2>>stderr.txt
done;

