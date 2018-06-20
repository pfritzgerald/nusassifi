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

import sys, re, string, os, operator, math, datetime, random, sqlite3
import common_params as cp
import specific_params as sp
import common_functions as cf 

MAX_INJ = sp.NUM_INJECTIONS
verbose = False
is_rf = False # RF=True, Inst=False

#################################################################
# Generate injection list of each
# 	- app
#   - instruction group 
#   - bit-flip model
#################################################################
def write_injection_list_file(app, igid, bfm, num_injections, total_count, countList):
	if verbose:
		print "total_count = %d, num_injections = %d" %(total_count, num_injections)
	fName = sp.app_log_dir[app] + "/injection-list/igid" + str(igid) + ".bfm" + str(bfm) + "." + str(num_injections) + ".txt"
	print fName
	f = open(fName, "w")

	while num_injections > 0 and  total_count != 0: # first two are kname and kcount
		num_injections -= 1
		injection_num = random.randint(0, total_count) # randomly select an injection index
		if igid == "rf":
			[inj_kname, inj_kcount, inj_icount] = cf.get_rf_injection_site_info(countList, injection_num) # convert injection index to [kname, kernel count, inst index]
			inj_op_id_seed = sp.num_regs[app][inj_kname]*random.random() # register selection
		else:
			[inj_kname, inj_kcount, inj_icount] = cf.get_injection_site_info(countList, injection_num, igid) # convert injection index to [kname, kernel count, inst index]
			inj_op_id_seed = random.random()
		inj_bid_seed = random.random() 
		selected_str = inj_kname + " " + str(inj_kcount) + " " + str(inj_icount) + " " + str(inj_op_id_seed) + " " + str(inj_bid_seed) + " "
		if verbose:
			print "%d/%d: Selected: %s" %(num_injections, total_count, selected_str)
		f.write(selected_str + "\n") # print injection site information
	f.close()

########################################################################
# FRITZ - write injection list file function for PC-specific injections
#####################################################################
def write_injection_list_file_pc(app, igid, bfm, pcList):
	#if verbose:
	#	print "total_count = %d, num_injections = %d" %(total_)
	fName = sp.app_log_dir[app] + "/injection-list/igid" + str(igid) + ".bfm" + str(bfm) + "." + str(sp.NUM_INJECTIONS)+ ".pc.txt"
	print fName
	f = open(fName, "w")
	total_faults = 0	
	for pupc in pcList:
		pc = pupc[0]
		pc_count = int(pupc[2])
		num_faults = int(round(sp.NUM_INJECTIONS * float(pupc[1])))
		if num_faults > (sp.NUM_INJECTIONS/100):
			num_faults=int(sp.NUM_INJECTIONS/100)
		print "PC: " + pc + " num faults: " + str(num_faults)
		for fault_id in range(0, num_faults):
			total_faults += 1 
			pc_count = random.randint(0, pc_count)
			inj_op_id_seed = random.random()
			inj_bid_seed = random.random()
			selected_str = pc + " " + str(pc_count)  + " " + str(inj_op_id_seed) + " " + str(inj_bid_seed) + " "
			#if verbose:
			#	print ""
			f.write(selected_str + "\n")
	f.close()	



########################################################################
# FRITZ - write injection list file function for interval-specific injections
#####################################################################
def write_injection_list_file_interval(app, igid, bfm, intervalList):
	#if verbose:
	#	print "total_count = %d, num_injections = %d" %(total_)
	fName = sp.app_log_dir[app] + "/injection-list/igid" + str(igid) + ".bfm" + str(bfm) + ".interval.txt"
	print fName
	f = open(fName, "w")
	total_faults = 0	
	num_faults_per_interval = intervalList[0][1]
	interval_size = intervalList[0][0]
	for i in range(0,len(intervalList)):
		num_igid_insts = intervalList[i][3]
		interval_id = intervalList[i][2]
		for fault_id in range(0, num_faults_per_interval):
			total_faults += 1 
			inst_id = random.randint(0, num_igid_insts)
			inj_op_id_seed = random.random()
			inj_bid_seed = random.random()
			selected_str = str(interval_size) + " " + str(interval_id) + " " +  str(inst_id) + " " + str(inj_op_id_seed) + " " + str(inj_bid_seed) + " "
			#if verbose:
			#	print ""
			f.write(selected_str + "\n")
	f.close()	

#################################################################
# Generate injection list of each app for 
# (1) RF AVF (for each error model)
# (2) instruction-level injections (for each error model and instruction type)
#################################################################
def gen_lists(app, countList, is_rf):
	if is_rf: # RF injection list
		total_count = cf.get_total_insts(countList)
		for bfm in sp.rf_bfm_list:
			write_injection_list_file(app, "rf", bfm, MAX_INJ, total_count, countList)
		
	else: # instruction injections
		total_icounts = cf.get_total_counts(countList)
		for igid in sp.igid_bfm_map:
			for bfm in sp.igid_bfm_map[igid]: 
				write_injection_list_file(app, igid, bfm, MAX_INJ, total_icounts[igid], countList)

###################################################################
# FRITZ - Generate injection list of each app for 
# instruction-level injections for each error model and instruction type
# for PC-based injections
#######################################################################
def gen_lists_pc(app, pcList):
	for igid in sp.igid_bfm_map:
		for bfm in sp.igid_bfm_map[igid]:
			write_injection_list_file_pc(app, igid, bfm, pcList)

####################################################################
# FRITZ added in for support for injections in specific PCs
# 	This could eventually be moved to common_functions.py
#
#	This function reads the interval file for an app and 
#		returns an array of this format:
#		[App, PC, PC_ratio]
#
######################################################################
def get_pc_distribution(app, db_file):
	pcList = []
	print app
	if not os.path.exists(db_file):
		print "%s DB file not found" %fName
		return pcList
	conn = sqlite3.connect(db_file)
	c = conn.cursor()
	pupcs = c.execute('SELECT PUPC, 1.0*(Weight)/IgIdMap.InstCount AS Pct, Weight FROM PUPCs,IgIdMap WHERE '\
			'Description LIKE \'DEST_REG\' AND IgIdMap.App==PUPCs.App AND IsDestReg==1 AND PUPCs.App IS '\
			'\'%s\' GROUP BY PUPC;' %(app)).fetchall()
	for pupc in pupcs:
		pcList.append([pupc[0],pupc[1],pupc[2]])
		print pupc[0],pupc[1]
	
	conn.close()
	#intervalList.append([interval_size, num_faults_per_interval, interval_id, num_igid_insts])

#	print "pclList : " + str(pcList)
	return pcList
	

###################################################################
# FRITZ - Generate injection list of each app for 
# instruction-level injections for each error model and instruction type
# for interval=specific injections
#######################################################################
def gen_lists_interval(app, pcList):
	for igid in sp.igid_bfm_map:
		for bfm in sp.igid_bfm_map[igid]:
			write_injection_list_file_interval(app, igid, bfm, intervalList)

####################################################################
# FRITZ added in for support for injections in specific intervals
# 	This should eventually be moved to common_functions.py
#
#	This function reads the interval file for an app and 
#		returns an array of this format:
#		[App, IntervalSize,NumFaultsPerInterval,IntervalList...]
#		IntervalList is a list of intervals that were selected for injections
#			by FIPoints algorithm
######################################################################
def read_interval_file(app_dir, app):
	intervalList = []
	fName = app_dir + "/interval.txt"
	if not os.path.exists(fName):
		print "%s file not found" %fName
		return intervalList
	f = open(fName, "r")
	next(f)
	next(f)	#skip the first two lines which contain 1.interval size for SASSI profiler and 2.format
	for i,line in enumerate(f):	#will probably contain only one line	
		line = line.rstrip().split(":")
		if i == 0:
			[interval_size, num_faults_per_interval] = [int(line[1]), int(line[2])]
			#intervalList.append([app,interval_size,num_faults_per_interval, interval1_id, interval2_id,....])
		else:
			[interval_id, num_igid_insts] = [int(line[0]), int(line[1])]
			intervalList.append([interval_size, num_faults_per_interval, interval_id, num_igid_insts])
	f.close()
	print "intervalList : " + str(intervalList)
	return intervalList
		
#################################################################
# Starting point of the script
#################################################################
def main():
	db_file = "profiling.db"
	if len(sys.argv) == 2: 
		is_rf = (sys.argv[1] == "rf")
		interval = False
		pc = False
	elif len(sys.argv) == 3:
		is_rf = (sys.argv[1] == "rf")
		interval = (sys.argv[2] == "interval")
		pc = (sys.argv[2] == "pc")
	else:
		print "Usage: ./script-name <rf or inst> [<pc or interval>]"
		print "There are two modes to conduct error injections"
		print "rf: tries to randomly pick a register and inject a bit flip in it (tries to model particle strikes in register file)"
		print "inst: tries to randomly pick a dynamic instruction and inject error in the destimation regsiter" 
		exit(1)
	
	# actual code that generates list per app is here
	for app in sp.apps:
		print "\nCreating list for %s ... " %(app)
		os.system("mkdir -p %s/injection-list" %sp.app_log_dir[app]) # create directory to store injection list
		
		if interval:
			intervalList = read_interval_file(sp.app_dir[app], app)
			if verbose: print intervalList
			gen_lists_interval(app, intervalList)
			print "OUTPUT : Check %s" % (sp.app_log_dir[app] + "/injection-list/")
		if pc:
			pcList = get_pc_distribution(app, db_file)
			if verbose: print pcList
			gen_lists_pc(app, pcList)

		else:	
			countList =  cf.read_inst_counts(sp.app_dir[app], app)
			total_count = cf.get_total_insts(countList)
			if total_count == 0:
				print "Something is not right. Total instruction count = 0\n";
				sys.exit(-1);
		
			gen_lists(app, countList, is_rf)
			print "Output: Check %s" %(sp.app_log_dir[app] + "/injection-list/")

if __name__ == "__main__":
    main()
