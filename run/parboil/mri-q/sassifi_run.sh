#echo "APP_DIR: ${APP_DIR}" 
${BIN_DIR}/mri-q -i ${APP_DIR}/../../datasets/mri-q/small/input/32_32_32_dataset.bin -o ${APP_DIR}/run/small/32_32_32_dataset.out  > stdout.txt 2>stderr.txt
