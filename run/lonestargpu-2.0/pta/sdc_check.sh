#!/bin/bash

touch diff.log
touch stderr_diff.log
#diff stderr.txt ${APP_DIR}/golden_stderr.txt > stderr_diff.log

grep Verifying stderr.txt | sed 's:.*txt::' > selected_output.txt
grep Verifying ${APP_DIR}/golden_stderr.txt | sed 's:.*txt::' > selected_golden_output.txt
diff selected_output.txt selected_golden_output.txt > stdout_diff.log

