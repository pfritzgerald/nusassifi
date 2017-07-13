trap "echo CAUGHT SIGNAL" SIGBUS SIGINT SIGSEGV SIGUSR1 SIGUSR2
#echo "THIS IS GAUSSIAN"
${BIN_DIR}/gaussian -f ${DATASET_DIR}/matrix16.txt >stdout.txt 2>stderr.txt
#${BIN_DIR}/gaussian -s 16 >>stdout.txt 2>>stderr.txt
