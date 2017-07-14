#!/bin/bash
python_app_dirs='
import specific_params as sp
for app in sp.apps:
  print sp.app_dir[app]
'
python_app_log_dirs='
import specific_params as sp
for app in sp.apps:
  print sp.app_log_dir[app]
'
where_to_run=$2
inst_rf="inst"
if [ "$inst_rf" == "inst" ] || [ "$inst_rf" == "rf" ] ; then
  printf "Mode: $inst_rf\n"
else
  inst_rf="inst"

fi
if [ "$where_to_run" == "cluster" ] || [ "$where_to_run" == "standalone" ] || [ "$where_to_run" == "multigpu" ] ; then
	printf "Running on $where_to_run\n"
else
	where_to_run="standalone"
fi

printf "Proceeding with $inst_rf on $where_to_run\n"
#printf "\nEnter directory for your application: "
app_dir_list=`python -c "$python_app_dirs"`
printf "\nDirectory list:\n" 
#set -x 

################################################
# Step 1: Set environment variables
################################################
printf "\n--------\nStep 1: Setting environment variables"
if [ `hostname -s` == "kepler1" ]; then
	export SASSIFI_HOME=/home/previlon/nusassifi/
	export SASSI_SRC=/home/previlon/SASSI/
	export INST_LIB_DIR=$SASSI_SRC/instlibs/lib/
	export CCDIR=/usr/bin/
	export CUDA_BASE_DIR=/home/previlon/sassi7/
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_BASE_DIR/lib64/:$CUDA_BASE_DIR/extras/CUPTI/lib64/
else
  . env.sh
fi

printf "\n------\nStep 1a: CLEAN?\n"
clean=$1
if [ "$clean" == "clean" ] ; then
  printf "Removing logs: \n"
  app_log_dir_list=`python -c "$python_app_log_dirs"`
#  printf "app log dir list: $app_log_dir_list"
  for app_log_dir in $app_log_dir_list
  do
    printf "\nRemoving from $app_log_dir ..."
    rm -rf $app_log_dir/*
    if [ $? -ne 0 ]; then
      echo "Could not clear app log directory: $app_log_dir"
      exit -1
    fi
    printf " ..Done"
  done
fi

for app_directory in $app_dir_list
do
  printf "\n--------------------------------\n$app_directory"
  printf "\n--------------------------------\n"

################################################
# Step 4.a: Build the app without instrumentation.
# Collect golden stdout and stderr files.
################################################
printf "\n-----------\nStep 4.1: Collect golden stdout.txt and stderr.txt files"
cd $app_directory
if [ $? -ne 0 ]; then
  echo "Problem with app directory"
  exit -1
fi
make 2> stderr.txt
make golden
if [ $? -ne 0 ]; then
    echo "Return code was not zero: $?"
		exit -1;
fi

################################################
# Step 5: Build the app for profiling and
# collect the instruction profile
################################################
printf "\n------------\nStep 5: Profile the application\n"
make OPTION=profiler
make test 
if [ $? -ne 0 ]; then
    echo "Return code was not zero: $?"
		exit -1;
fi

################################################
# Step 6: Build the app for error injectors
################################################
printf "\n-------------\nStep 6: Prepare application for error injection\n"
make OPTION=inst_injector

done
################################################
# Step 7.b: Generate injection list for the 
# selected error injection model
################################################
printf "\n------------\nStep 7.2: Generate injection list for instruction-level error injections\n"
#cd -
cd $SASSIFI_HOME/scripts/
python generate_injection_list.py $inst_rf 
if [ $? -ne 0 ]; then
    echo "Return code was not zero: $?"
		exit -1;
fi

################################################
# Step 8: Run the error injection campaign 
################################################
printf "\n------------\nStep 8: Run the error injection campaign\n"
python run_injections.py $inst_rf $where_to_run
# to run the injection campaign on a single machine with single gpu

#python run_injections.py $inst_rf multigpu 
# to run the injection campaign on a single machine with multiple gpus. 

################################################
# Step 9: Parse the results
################################################
#printf "\nStep 9: Parse results\n"
#python parse_results.py $inst_rf

