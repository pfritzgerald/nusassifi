#!/bin/bash

touch diff.log

diff stderr.txt ${APP_DIR}/golden_stderr.txt > stderr_diff.log


grep result stdout.txt | sed 's:.*result::' > selected_output.txt
grep result ${APP_DIR}/golden_stdout.txt | sed 's:.*result::' > selected_golden_output.txt
diff -B selected_output.txt selected_golden_output.txt > stdout_diff.log

