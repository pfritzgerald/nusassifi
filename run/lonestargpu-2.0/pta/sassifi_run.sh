IPFX=${APP_DIR}/../../inputs/ex

NODE=${IPFX}_nodes.txt
CONS=${IPFX}_constraints_after_hcd.txt
HCD=${IPFX}_hcd.txt
SOLN=${IPFX}_correct_soln_001.txt


[ -f "$NODE" ] || echo $NODE does not exist
[ -f "$CONS" ] || echo $CONS does not exist
[ -f "$HCD" ] || echo $HCD does not exist
[ -f "$SOLN" ] || echo $SOLN does not exist

if [ -f "$NODE" ] && [ -f "$CONS" ] && [ -f "$HCD" ] && [ -f "$SOLN" ]; then
#    echo ${BIN_DIR}/pta $NODE $CONS $HCD $SOLN 1 1
    ${BIN_DIR}/pta $NODE $CONS $HCD $SOLN 1 1 >stdout.txt 2>stderr.txt
fi;


