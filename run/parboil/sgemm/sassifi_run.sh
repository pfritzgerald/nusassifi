#echo "APP_DIR: ${APP_DIR}" 
${BIN_DIR}/sgemm -i ${APP_DIR}/../../datasets/sgemm/medium/input/matrix1.txt,${APP_DIR}/../../datasets/sgemm/medium/input/matrix2t.txt,${APP_DIR}/../../datasets/sgemm/medium/input/matrix2t.txt -o ${APP_DIR}/run/medium/matrix3.txt  > stdout.txt 2>stderr.txt
