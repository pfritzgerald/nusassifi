################################################################################### 
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###################################################################################

import os, sys, re, string, operator, math, datetime, subprocess, time, multiprocessing, pkgutil
import common_params as cp
import specific_params as sp

###############################################################################
# Basic functions and parameters
###############################################################################
before = -1

def print_usage():
	print "Usage: \n run_injections.py rf/inst standalone/multigpu/cluster <clean>"
	print "Example1: \"run_injections.py rf standalone\" to run jobs on the current system"
	print "Example1: \"run_injections.py inst multigpu\" to run jobs on the current system using multiple gpus"
	print "Example2: \"run_injections.py inst cluster clean\" to launch jobs on cluster and clean all previous logs/results"

############################################################################
# Print progress every 10 minutes for jobs submitted to the cluster
############################################################################
def print_heart_beat(nj, where_to_run):
	if where_to_run != "cluster":
		return
	global before
	if before == -1:
		before = datetime.datetime.now()
	if (datetime.datetime.now()-before).seconds >= 10*60:
		print "Jobs so far: %d" %nj
		before = datetime.datetime.now()

def get_log_name(app, igid, bfm):
	return sp.app_log_dir[app] + "results-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(sp.NUM_INJECTIONS) + ".txt"

############################################################################
# Clear log conent. Default is to append, but if the user requests to clear
# old logs, use this function.
############################################################################
def clear_results_file(app):
	for bfm in sp.rf_bfm_list: 
		open(get_log_name(app, "rf", bfm)).close()
	for igid in sp.igid_bfm_map:
		for bfm in sp.igid_bfm_map[igid]:
			open(get_log_name(app, igid, bfm)).close()

############################################################################
# count how many jobs are done
############################################################################
def count_done(fname):
	return sum(1 for line in open(fname)) # count line in fname 


############################################################################
# check queue and launch multiple jobs on a cluster 
# This feature is not implemented.
############################################################################
def create_sbatch_script(app,array_num):
	filename =  sp.SASSIFI_HOME + "/scripts/tmp/" + app + "/" + str(array_num)+".sbatch"
	user = os.environ["USER"]
	outf = open(filename, "w")
	outf.write("#!/bin/bash\n"
		"# sassifi.sbatch\n#\n"
		"#SBATCH --exclusive\n"
		"#SBATCH -J " + app + str(array_num) + "\n"
		"#SBATCH -p par-gpu\n"
		"#SBATCH -n 32\n"
		"#SBATCH -N 1\n"
		"#SBATCH -x compute-2-133\,compute-2-148\n"
		"#SBATCH -o /gss_gpfs_scratch/" + user + "/nusassifi/" + app + "/" + app + "_sbatch_%j.out\n"
		"#SBATCH -e /gss_gpfs_scratch/" + user + "/nusassifi/" + app + "/" + app + "_sbatch_%j.err\n\n")
	if sp.USE_ARRAY:
		outf.write("cmd=`sed \"${SLURM_ARRAY_TASK_ID}q;d\" cmds_" + str(array_num) + ".out`\n"			
		"$cmd\n")
	else:
		outf.write("while IFS= read line\ndo\n"
		"$line\n"
		"done < cmds_" + str(array_num) + ".out\n")
	outf.close()
	return filename

def check_and_submit_cluster(cmd, app, total_jobs, interval_mode, pc_mode):
	threshold_num = 1000
	array_num = ((total_jobs - 1) / threshold_num) + 1
	if total_jobs < sp.THRESHOLD_JOBS + 1:
		os.system("echo " + cmd + " >> " + sp.SASSIFI_HOME + "/scripts/tmp/" + app + "/cmds_" + str(array_num) + ".out")
	if (total_jobs == sp.THRESHOLD_JOBS) or ((total_jobs % threshold_num) == 0)\
			or (interval_mode and (total_jobs == total_interval_jobs))\
			or (pc_mode and (total_pc_jobs == total_jobs)):
		sbatch_script=create_sbatch_script(app,array_num)
		num_jobs = (total_jobs % threshold_num) if (total_jobs % threshold_num) else threshold_num
		if sp.USE_ARRAY:
			os.system("sbatch -D " + sp.SASSIFI_HOME + "/scripts/tmp/" + app + " --array=1-" + str(num_jobs) + " " + sbatch_script)
		else:
			os.system("sbatch -D " + sp.SASSIFI_HOME + "/scripts/tmp/" + app + " " + sbatch_script)
#		os.system("rm cmds.out")

############################################################################
# check queue and launch multiple jobs on the multigpu system 
############################################################################
jobs_list = []
pool = multiprocessing.Pool(sp.NUM_GPUS) # create a pool

def check_and_submit_multigpu(cmd):
	if len(jobs_list) == sp.NUM_GPUS:
		pool.map(os.system, jobs_list) # launch jobs in parallel
		del jobs_list[:] # clear the list
	else:
		jobs_list.append("CUDA_VISIBLE_DEVICES=" + str(len(jobs_list)) + " " + cmd)
		# print "appending.. " 


###############################################################################
# Run Multiple injection experiments
###############################################################################
def run_multiple_injections_igid(app, is_rf, igid, where_to_run, interval_mode, pc_mode):
	bfm_list = sp.rf_bfm_list if is_rf else sp.igid_bfm_map[igid]
	if where_to_run == "cluster":
		# Create directories to store logs and tmp files for slurm
		if not os.path.isdir(sp.SASSIFI_HOME + "/scripts/tmp/" + app):
			os.system("mkdir -p " + sp.SASSIFI_HOME + "/scripts/tmp/" + app)
		user = os.environ["USER"]
		if not os.path.isdir("/gss_gpfs_scratch/" + user + "/nusassifi/" + app):
			os.system("mkdir -p /gss_gpfs_scratch/" + user + "/nusassifi/" + app)
		os.system("rm -f " + sp.SASSIFI_HOME + "/scripts/tmp/" + app + "/cmds_*.out") 
	for bfm in bfm_list:
		#print "App: %s, IGID: %s, EM: %s" %(app, cp.IGID_STR[igid], cp.EM_STR[bfm])
		total_jobs = 0
		if interval_mode:
			inj_list_filenmae = sp.app_log_dir[app] + "/injection-list/igid" + str(igid) + ".bfm" + str(bfm) + ".interval.txt"
			global total_interval_jobs
			interval_file = open(sp.app_dir[app]+"/interval.txt", "r")
			next(interval_file)
			next(interval_file)
			interval_line = next(interval_file).split(":")
			total_interval_jobs = int(interval_line[2]) * len(interval_line[3:])
		elif pc_mode:
			inj_list_filenmae = sp.app_log_dir[app] + "/injection-list/igid" + str(igid) + ".bfm" + str(bfm) + "." + \
					str(sp.NUM_INJECTIONS) + ".pc.txt"
			global total_pc_jobs
			total_pc_jobs = sum(1 for line in open(inj_list_filenmae))
		else:
			inj_list_filenmae = sp.app_log_dir[app] + "/injection-list/igid" + str(igid) + ".bfm" + str(bfm) + "." + str(sp.NUM_INJECTIONS) + ".txt"
		inf = open(inj_list_filenmae, "r")
		
		for line in inf: # for each injection site 
			total_jobs += 1
			if not interval_mode and total_jobs > sp.THRESHOLD_JOBS: 
				break; # no need to run more jobs

			#_Z24bpnn_adjust_weights_cudaPfiS_iS_S_ 0 1297034 0.877316323856 0.214340876321
			if len(line.split()) >= 3: 
				if interval_mode:
					[interval_size, interval_id, iid, opid, bid] = line.split()
					if cp.verbose: print "\n%d: app:%s, interval_mode, igid=%d, bfm=%d"\
					"intervalSize=%s, intervalID=%s, instID=%s, opId=%s, bitLocation=%s"\
						%(total_jobs, app, igid, bfm, interval_size, interval_id, iid, opid, bid)
					cmd = "%s %s/scripts/run_one_injection.py interval_mode %s %s %s %s %s %s %s %s"\
						% (cp.PYTHON_P, sp.SASSIFI_HOME, str(igid), str(bfm), app, interval_size, interval_id, iid, opid, bid)
				elif pc_mode:
					[pc, pc_count, opid, bid] = line.split()
					if cp.verbose: print "\n%d: app=%s, pc_mode, igid=%s, bfm=%d, "\
							"PC=%s, PC_count=%s, opId=%s, bitLocation=%s"\
							%(total_jobs, app, igid, bfm, pc, pc_count, opid, bid)
					cmd = "%s %s/scripts/run_one_injection.py pc_mode %s %s %s %s %s %s %s" %(cp.PYTHON_P,\
							sp.SASSIFI_HOME, str(igid), str(bfm), app, pc, pc_count, opid, bid)
				else:
					[kname, kcount, iid, opid, bid] = line.split() # obtains params for this injection
					if cp.verbose: print "\n%d: app=%s, Kernel=%s,"\
					"kcount=%s, igid=%d, bfm=%d, instID=%s,"\
					"opID=%s, bitLocation=%s" %(total_jobs, app, kname, kcount, igid, bfm, iid, opid, bid)
					cmd = "%s %s/scripts/run_one_injection.py normal_mode %s %s %s %s %s %s %s %s" %(cp.PYTHON_P,
							sp.SASSIFI_HOME, str(igid), str(bfm), app, kname, kcount, iid, opid, bid)
				if where_to_run == "cluster":
					check_and_submit_cluster(cmd, app, total_jobs, interval_mode,pc_mode)
				elif where_to_run == "multigpu":
					check_and_submit_multigpu(cmd)
				else:
					os.system(cmd)
				if cp.verbose: print "done injection run "
			else:
				print "Line doesn't have enough params:%s" %line
			print_heart_beat(total_jobs, where_to_run)
#	if where_to_run == "cluster":
#		os.system("cat " + app + "_sbatch_*.out >" + app + "_sbatch.out")
#		os.system("cat " + app + "_sbatch_*.err >" + app + "_sbatch.err")
#		os.system("rm " + app + "_sbatch_*.* sassifi.sbatch")




###############################################################################
# wrapper function to call either RF injections or instruction level injections
###############################################################################
def run_multiple_injections(app, is_rf, where_to_run, interval_mode, pc_mode):
	if is_rf:
		run_multiple_injections_igid(app, is_rf, "rf", where_to_run, interval_mode, pc_mode)
	else:
		for igid in sp.igid_bfm_map:
			run_multiple_injections_igid(app, is_rf, igid, where_to_run, interval_mode, pc_mode)

###############################################################################
# Starting point of the execution
###############################################################################
def main(): 
	if len(sys.argv) >= 3: 
		where_to_run = sys.argv[2]
		interval_mode = False
		pc_mode = False
		if len(sys.argv) == 4:
			interval_mode = (sys.argv[3] == "interval")
			pc_mode = (sys.argv[3] == "pc")
		if len(sys.argv) == 5:
			interval_mode |= (sys.argv[4] == "interval")
			pc_mode |= (sys.argv[4] == "pc")
		if where_to_run != "standalone":
			if pkgutil.find_loader('lockfile') is None:
				print "lockfile module not found. This python module is needed to run injection experiments in parallel." 
				sys.exit(-1)
	
		sorted_apps = [app for app, value in sorted(sp.apps.items(), key=lambda e: e[1][2])] # sort apps according to expected runtimes
		for app in sorted_apps: 
		 	print app
			if not os.path.isdir(sp.app_log_dir[app]): os.system("mkdir -p " + sp.app_log_dir[app]) # create directory to store summary
			if len(sys.argv) == 4: 
				if sys.argv[3] == "clean":
					clear_results_file(app) # clean log files only if asked for
	
		 	run_multiple_injections(app, (sys.argv[1] == "rf"), where_to_run, interval_mode, pc_mode)
	
	else:
		print_usage()

if __name__ == "__main__":
    main()
