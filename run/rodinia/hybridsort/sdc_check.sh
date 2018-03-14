#!/bin/bash
# if your program creates an output file (e.g., output.txt) compare it to the file created just now and store the difference in diff.log
#diff result.txt ${APP_DIR}/result.txt > diff.log
touch diff.log stdout_diff.log stderr_diff.log 
# comparing stdout generated by your program
# comparing stderr generated by your program
diff stderr.txt ${APP_DIR}/golden_stderr.txt > stderr_diff.log
# Application specific output: The following check will be performed only if at least one of diff.log, stdout_diff.log, and stderr_diff.log is different
# sed 's/:::Injecting.*::://g' stdout.txt | sed '/Time to generate/d' | sed '/Encoding time/d' > selected_output.txt 
# sed '/Time to generate/d' ${APP_DIR}/golden_stdout.txt | sed '/Encoding time/d' > selected_golden_output.txt 
if ! grep -q "PASSED" stdout.txt; then
	cp stdout.txt stdout_diff.log
fi
#diff selected_output.txt selected_golden_output.txt > stdout_diff.log

