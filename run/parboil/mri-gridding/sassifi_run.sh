#echo "APP_DIR: ${APP_DIR}" 
${BIN_DIR}/mri-gridding -i ${APP_DIR}/../../datasets/mri-gridding/small/input/small.uks -o ${APP_DIR}/run/small/output.txt -- 32 0  > stdout.txt 2>stderr.txt
