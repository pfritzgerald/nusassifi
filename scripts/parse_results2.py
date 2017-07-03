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


import os, sys, re, string, math, datetime, time, pkgutil
from optparse import OptionParser
import common_params as cp
import specific_params as sp
import common_functions as cf 
import sqlite3
import matplotlib.pyplot as plt
import numpy as np


results_app_table = {} # app, igid, bfm, outcome, 

inj_types = ["inst","rf"]


###############################################################################
# inst_fraction contains the fraction of IADD, FADD, IMAD, FFMA, ISETP, etc. 
# instructions per application
###############################################################################
inst_fraction = {}
inst_count = {}

def parse_results_file(app, igid, bfm, c):
	try:
		rf = open(sp.app_log_dir[app] + "results-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(sp.NUM_INJECTIONS) + ".txt", "r")
	except IOError: 
		print "app=%s, igid=%d, bfm=%d " %(app, igid, bfm),
		print sp.app_log_dir[app] + "results-igid" + str(igid) + ".bfm" + str(bfm) + "." + str(sp.NUM_INJECTIONS) + ".txt"
		return 
	suite = sp.apps[app][0]
        print "file is " + rf.name
	num_lines = 0
	for line in rf: # for each injection site 
		#Example line: _Z22bpnn_layerforward_CUDAPfS_S_S_ii-0-26605491-0.506809798834-0.560204950825:..:MOV:773546:17:0.759537:3:dmesg, 
		#kname-kcount-iid-allIId-opid-bid:pc:opcode:tid:injBID:runtime_sec:outcome_category:dmesg
		words = line.split(":")
		inj_site_info = words[0].split("-")
		[job_id, kname, invocation_index, opcode, injBID, runtime, outcome] = \
			[int(inj_site_info[0]), inj_site_info[1], int(inj_site_info[2]), words[5], int(words[7]), float(words[8]), int(words[9])]
		inst_id = int(inj_site_info[3])
		opIdSeed = inj_site_info[4]
		bIdSeed = inj_site_info[5]
#		print "words[1]: "+ str(words[1]),
		pc_text = '0x'+str(words[1])
		pc_count = int(words[2])
                bb_id = int(words[3])
                global_inst_id = int(words[4])
		if pc_text == '0x':
			pc_text = "0x0"
#		print "PC text: "  + " => " + pc_text
#		pc = int(pc_text,0)
		tId = int(words[6])
		c.execute('INSERT OR IGNORE INTO Results '\
            'VALUES(NULL, %d, \'%s\',\'%s\', \'%s\', \'%s\', \'%s\', %d, %d, %d, %d, \'%s\', %d, %d, %d, \'%s\', %d, %d, %f, %d)'     
            %(job_id,suite,app, kname, opIdSeed, bIdSeed, igid, bfm, invocation_index, inst_id, pc_text,
                pc_count, bb_id, global_inst_id, opcode, tId, injBID, runtime, (outcome-1)))

		num_lines += 1
	rf.close()

	if num_lines == 0 and app in results_app_table and os.stat(sp.app_log_dir[app] + "injection-list/igid" + str(igid) + ".bfm" + str(bfm) + "." + str(sp.NUM_INJECTIONS) + ".txt").st_size != 0: 
		print "%s, igid=%d, bfm=%d not done" %(app, igid, bfm)

###################################################################################
# Parse results files and populate summary to results table 
###################################################################################
def parse_results_apps(typ,c): 
	for app in sp.parse_apps:
                print app
		if typ == "inst":
			for igid in sp.parse_igid_bfm_map:
				for bfm in sp.parse_igid_bfm_map[igid]:
					parse_results_file(app, igid, bfm, c)
		else:
			for bfm in sp.parse_rf_bfm_list:
				parse_results_file(app, "rf", bfm, c)


def parse_options():
	parser = OptionParser()
	parser.add_option("-t", "--type", dest="inj_type",
				help="Injection Type <inst/rf>", metavar="INJ_TYPE")
	parser.add_option("-d", "--database", dest="database_file",
                  help="Database file where our data is")
	parser.add_option("-a", "--app", dest="application",
                  help="Application to analyze")
	
	# Create a database if one was not passed.
	(options, args) = parser.parse_args()
	if options.inj_type:
		if options.inj_type not in inj_types:
			parser.error("inj_type should be one of: %s - provided:%s"
			% (inj_types,options.inj_type))
	else:
		options.inj_type = "inst"
	if not options.database_file:
		options.database_file = "data.db"
		
	return options.database_file, options.inj_type, options.application

def print_usage():
	print "Usage: \n python parse_results.py rf/inst"
	exit(1)

def CreateNewDB(c):
	print "creating data DB"
	c.execute('CREATE TABLE IF NOT EXISTS '\
          'Results(ID INTEGER PRIMARY KEY, JobId INTEGER, Suite TEXT, App TEXT,  kName TEXT, '\
	  'OpIdSeed TEXT, BIDSeed TEXT, IgId INTEGER, '\
          'BFM INTEGER, InvocationIdx INTEGER, InstId INTERGER, PC TEXT, PCCount INTEGER, BBId '\
          'INTEGER, GlobalInstId INTEGER, '\
          'Opcode TEXT, TId INTEGER, InjBId INTEGER, Runtime INTEGER, OutcomeID INTEGER)')
	c.execute('CREATE TABLE IF NOT EXISTS '\
          'OutcomeMap(ID INTEGER PRIMARY KEY, Description TEXT)')
	c.execute('CREATE TABLE IF NOT EXISTS '\
          'IgIdMap(ID INTEGER PRIMARY KEY, IDNum INTEGER, Description TEXT, App TEXT, InstCount INTEGER)')
	c.execute('CREATE TABLE IF NOT EXISTS '\
          'BFMMap(ID INTEGER PRIMARY KEY, IDNum INTEGER, Description TEXT, App TEXT)')
	c.execute('CREATE TABLE IF NOT EXISTS '\
          'OpcodeMap(ID INTEGER PRIMARY KEY, Description TEXT, App TEXT, InstCount INTEGER)')	
	c.execute('CREATE TABLE IF NOT EXISTS '\
		'Kernels(ID INTEGER PRIMARY KEY, Application TEXT, KernelName TEXT, '\
		'InvocationIdx INTEGER, InvInstCount INTEGER, AppInstCount INTEGER)')

	######
	# fill up OutcomeMap table
	#########	
	for cat in range(cp.NUM_CATS-1):
#		print "cat %d cat_str %s " % (cat, cp.CAT_STR[cat])
		c.execute('INSERT OR IGNORE INTO OutcomeMap '\
		'VALUES(%d, \'%s\')' % (cat, cp.CAT_STR[cat]))
	
	##########
	# Filling up IgIdMap
	#########
	for app in sp.apps:
		countList = cf.read_inst_counts(sp.app_dir[app],app)
		#print countList
		for igid in range(cp.NUM_INST_TYPES):
			igid_inst_count = 0
			for l in countList:
				igid_inst_count += int(l[igid+2])
			c.execute('INSERT OR IGNORE INTO IgIdMap '\
					'VALUES(NULL, %d, \'%s\', \'%s\',%d)' % (igid,cp.IGID_STR[igid], app, igid_inst_count))
	##########
	# Filling up BitFlipModelMap (BFMMap)
	#########
	for app in sp.apps:
		countList = cf.read_inst_counts(sp.app_dir[app],app)
		#print countList
		for bfm in range(len(cp.EM_STR)):
			c.execute('INSERT OR IGNORE INTO BFMMap '\
					'VALUES(NULL, %d, \'%s\', \'%s\')'
                                        %(bfm,cp.EM_STR[bfm], app))
				
	###########
	# Filling up OpcodeMap
	###########
	opcode_list_str = "ATOM:ATOMS:B2R:BAR:BFE:BFI:BPT:BRA:BRK:BRX:CAL:CAS:CCTL:CCTLL:CCTLT:CONT:CS2R:CSET:CSETP:DADD:DEPBAR:DFMA:DMNMX:DMUL:DSET:DSETP:EXIT:F2F:F2I:FADD:FADD32I:FCHK:FCMP:FFMA:FFMA32I:FLO:FMNMX:FMUL:FMUL32I:FSET:FSETP:FSWZ:FSWZADD:I2F:I2I:IADD:IADD3:IADD32I:ICMP:IMAD:IMAD32I:IMADSP:IMNMX:IMUL:IMUL32I:ISAD:ISCADD:ISCADD32I:ISET:ISETP:JCAL:JMX:LD:LDC:LDG:LDL:LDLK:LDS:LDSLK:LDS_LDU:LDU:LD_LDU:LEA:LEPC:LONGJMP:LOP:LOP3:LOP32I:MEMBAR:MOV:MUFU:NOP:P2R:PBK:PCNT:PEXIT:PLONGJMP:POPC:PRET:PRMT:PSET:PSETP:R2B:R2P:RED:RET:RRO:S2R:SEL:SHF:SHFL:SHL:SHR:SSY:ST:STG:STL:STS:STSCUL:STSUL:STUL:SUATOM:SUBFM:SUCLAMP:SUEAU:SULD:SULDGA:SULEA:SUQ:SURED:SUST:SUSTGA:SYNC:TEX:TEXDEPBAR:TEXS:TLD:TLD4:TLD4S:TLDS:TXQ:UNMAPPED:USER_DEFINED:VMNMX:VOTE:XMAD"
	opcode_list = opcode_list_str.split(":")
#	print "OPCODE LIST: " + str(opcode_list)

	for app in sp.apps:
		countList = cf.read_inst_counts(sp.app_dir[app], app)
		total_count = cf.get_total_counts(countList)
		for i in range(len(opcode_list)):
			c.execute('INSERT OR IGNORE INTO OpcodeMap '\
					'VALUES(NULL, \'%s\', \'%s\',%d)' %(opcode_list[i], app, total_count[i+cp.NUM_INST_TYPES+1]))
	#	print "len total counts " + str(len(total_count))
	#	print "len opcode_list: " + str(len(opcode_list))

	for app in sp.apps:
#		print "App: " + app
		countList =  cf.read_inst_counts(sp.app_dir[app], app)
		#print "countList: " + str(countList)
		for l in countList:
			total_inst_count = 0
			for i in range(cp.NUM_INST_TYPES+3, len(countList[0])): # 3: 1 for kname, 1 for kcount and 1 for WILL NOT EXECUTE instruction count
				total_inst_count += int(l[i])
			kernel_name = str(l[0]) 
			invocation_idx = int(l[1])
			app_inst_count = cf.get_total_insts(countList)
			
			c.execute('INSERT OR IGNORE INTO Kernels '\
			'VALUES(NULL, \'%s\',\'%s\', %d, %d, %d)'
			% (app, kernel_name, invocation_idx, total_inst_count, app_inst_count))			


###############################################################################
# Main function that processes files, analyzes results and prints them to an
# xlsx file
###############################################################################
def main():
	db_file, inj_type, application = parse_options()

	print "DB file is : " + db_file
	conn = sqlite3.connect(db_file)
	c = conn.cursor()
	if db_file == "data.db":
		CreateNewDB(c)		
	#		total_count = cf.get_total_insts(countList)
		parse_results_apps(inj_type, c) # parse sassifi results into local data structures
	conn.commit()
	conn.close()

if __name__ == "__main__":
    main()
