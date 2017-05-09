#echo "APP_DIR: ${APP_DIR}" 
${BIN_DIR}/stencil -i \
  ${APP_DIR}/../../datasets/stencil/default/input/512x512x64x100.bin -o \
  ${APP_DIR}/run/default/512x512x64.out -- 512 512 64 100 > stdout.txt 2>stderr.txt
