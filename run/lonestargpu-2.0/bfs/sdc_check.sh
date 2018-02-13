#!/bin/bash

touch diff.log

diff stderr.txt ${APP_DIR}/golden_stderr.txt > stderr_diff.log

diff ${APP_DIR}/bfs-USA-road-d.FLA.gr-bfs-output.txt bfs-USA-road-d.FLA.gr-bfs-output.txt > diff.log
touch  stdout_diff.log

