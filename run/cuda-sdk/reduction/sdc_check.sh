#!/bin/bash
# if your program creates an output file (e.g., output.txt) compare it to the file created just now and store the difference in diff.log
touch diff.log
#./reduction > pass.txt
if ! grep -q "Test passed" stdout.txt; then
  cp stdout.txt diff.log
fi
##################################
#diff  <(sed 's/:::Injecting.*::://g' stdout.txt) ${APP_DIR}/golden_stdout.txt > stdout_diff.log
touch stdout_diff.log
# comparing stderr generated by your program
touch stderr_diff.log
# Application specific output: The following check will be performed only if at least one of diff.log, stdout_diff.log, and stderr_diff.log is different
#grep "Checking computed" stdout.txt > selected_output.txt 
#grep "Checking computed" ${APP_DIR}/golden_stdout.txt > selected_golden_output.txt 
#diff selected_output.txt selected_golden_output.txt > special_check.log
