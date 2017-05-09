#echo "APP_DIR: ${APP_DIR}" 
${BIN_DIR}/lbm -i ${APP_DIR}/../../datasets/lbm/short/input/120_120_150_ldc.of -o ${APP_DIR}/run/short/reference.dat -- 100  > stdout.txt 2>stderr.txt
