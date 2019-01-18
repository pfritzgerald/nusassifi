import sqlite3
import sys
import specific_params as sp

# threshold=0.1 * sp.NUM_INJECTIONS
verbose = False

def getAppOutcomes(app, c, threshold):
	print "\n----------------------------------------------------------------"
	print app
	num_masked = 0
	num_dues = 0
	num_sdcs = 0
	num_faults = 0
	num_faults_trad = 0
	num_unknwn = 0
	num_filtered_faults = 0
	common_pcs = c.execute('SELECT Results.PC, COUNT(Results.ID) FROM Results '\
			'WHERE App IS \'%s\' GROUP BY Results.PC HAVING COUNT(Results.ID) '\
			' >=	%d'
			% (app, threshold)).fetchall()
	if verbose: print "common pcs: " + str(common_pcs)
	for pc in common_pcs:
		if pc[0] == '0x0':
			continue
		pc_value = pc[0]
		num_pc_faults = pc[1]
		num_faults += num_pc_faults
		num_filtered_faults += threshold
#		print 'SELECT 1.0*(Weight)/IgIdMap.InstCount AS Pct FROM PUPCs,IgIdMap WHERE '\
#			'Description LIKE \'DEST_REG\' AND IgIdMap.App==PUPCs.App AND IsDestReg==1 AND PUPCs.App IS \'%s\' '\
#			'AND PUPC IS \'%s\';' %(app, pc_value)
		num_pc_faults_trad = c.execute('SELECT 1.0*(Weight)/IgIdMap.InstCount AS Pct FROM PUPCs,IgIdMap WHERE '\
			'Description LIKE \'DEST_REG\' AND IgIdMap.App==PUPCs.App AND IsDestReg==1 AND PUPCs.App IS \'%s\' '\
			'AND PUPC IS \'%s\';' 
			%(app, pc_value)).fetchone()[0] * sp.NUM_INJECTIONS
#		print "PC: " + pc_value + " - num_pc_faults_trad: " + str(num_pc_faults_trad)
#		print "\tPC: " + str(pc_value) + " num Faults: " + str(num_pc_faults)
		num_faults_trad += num_pc_faults_trad
		num_masked += c.execute('SELECT COUNT(Results.ID),PC,OutcomeID from OutcomeMap, '\
				'Results WHERE App IS \'%s\' '\
				'AND PC IS \'%s\' AND OutcomeID==OutcomeMap.ID '\
				'AND Description LIKE \'Masked:%%\';'
				% (app, pc_value)).fetchone()[0]\
									*(float(num_pc_faults_trad)/threshold)
		num_dues += c.execute('SELECT COUNT(Results.ID),PC,OutcomeID from OutcomeMap, '\
				'Results WHERE App IS \'%s\' '\
				'AND PC IS \'%s\' AND OutcomeID==OutcomeMap.ID '\
				'AND Description LIKE \'%%DUE:%%\';'
				% (app, pc_value)).fetchone()[0] \
									* (float(num_pc_faults_trad)/threshold)
		num_sdcs += c.execute('SELECT COUNT(Results.ID),PC,OutcomeID from OutcomeMap, '\
				'Results WHERE App IS \'%s\' '\
				'AND PC IS \'%s\' AND OutcomeID==OutcomeMap.ID '\
				'AND Description LIKE \'SDC:%%\';'
				% (app, pc_value)).fetchone()[0] \
						* (float(num_pc_faults_trad)/threshold)
		num_unknwn += c.execute('SELECT COUNT(Results.ID),PC,OutcomeID FROM OutcomeMap, '\
				'Results WHERE App IS \'%s\' '\
				'AND PC IS \'%s\' AND OutcomeID==OutcomeMap.ID '\
				'AND Description LIKE \'Uncategorized%%\';'
				% (app, pc_value)).fetchone()[0] \
						* (float(num_pc_faults_trad)/threshold)
	num_remained = c.execute('SELECT TOTAL(R.c) FROM '\
			'(SELECT COUNT(ID) as c,PC,OutcomeID FROM Results WHERE App IS '\
			'\'%s\'	GROUP BY PC HAVING COUNT(ID)<%d) AS R;'
			% (app,threshold)).fetchone()[0]
	num_filtered_faults += num_remained
	num_faults += num_remained
	num_faults_trad += num_remained
	num_masked += c.execute('SELECT TOTAL(R.c) FROM OutcomeMap, '\
			'(SELECT COUNT(ID) as c,PC,OutcomeID FROM Results WHERE App IS '\
			'\'%s\'	GROUP BY PC HAVING COUNT(ID)<%d) AS R WHERE OutcomeMap.ID '\
			'= R.OutcomeID AND Description LIKE \'Masked:%%\';'
			% (app,threshold)).fetchone()[0]
	num_dues += c.execute('SELECT TOTAL(R.c) FROM OutcomeMap, '\
		'(SELECT COUNT(ID) as c,PC,OutcomeID FROM Results WHERE App IS '\
		'\'%s\'	GROUP BY PC HAVING COUNT(ID)<%d) AS R WHERE OutcomeMap.ID '\
		'= R.OutcomeID AND Description LIKE \'%%DUE:%%\';'
		% (app,threshold)).fetchone()[0]
	num_sdcs += c.execute('SELECT TOTAL(R.c) FROM OutcomeMap, '\
		'(SELECT COUNT(ID) as c,PC,OutcomeID FROM Results WHERE App IS '\
		'\'%s\'	GROUP BY PC HAVING COUNT(ID)<%d) AS R WHERE OutcomeMap.ID '\
		'= R.OutcomeID AND Description LIKE \'SDC:%%\';'
		% (app,threshold)).fetchone()[0]
	num_unknwn += c.execute('SELECT TOTAL(R.c) FROM OutcomeMap, '\
			'(SELECT COUNT(ID) as c,PC,OutcomeID FROM Results WHERE App IS '\
			'\'%s\'	GROUP BY PC HAVING COUNT(ID)<%d) AS R WHERE OutcomeMap.ID '\
			'= R.OutcomeID AND Description LIKE \'Uncategorized%%\';'
			% (app,threshold)).fetchone()[0]

	print "\tNumFaults: " + str(num_faults) + " - Filtered: " + \
			str(num_filtered_faults) + ", Trad: " + str(num_faults_trad) + ", Masked:" + str(num_masked)\
			+ ", SDCs:" + str(num_sdcs) + ", DUES:" + str(num_dues)	+", Unknwn:" + str(num_unknwn)
	return float(num_masked)/num_faults_trad, float(num_dues)/num_faults_trad, \
			float(num_sdcs)/num_faults_trad, float(num_unknwn)/num_faults_trad, num_filtered_faults

def getFilteredOutcomes(c):
	masked_list = []
	due_list = []
	pot_due_list = []
	sdc_list = []
	num_interval_list = []
	threshold = sp.NUM_INJECTIONS * 0.01
	app_list = c.execute('SELECT App FROM Results GROUP BY App;').fetchall()
	app_list = [row[0] for row in app_list]
#	app_list = ['gaussian']
	for app in app_list:
#		print app
		num_masked,num_dues,num_sdcs,num_unknwn,num_filtered_faults = getAppOutcomes(app, c, threshold)
		print "\tMASKED_PCT: " + str(num_masked) + " -- SDC_PCT: " + str(num_sdcs)\
				+ " -- DUE_PCT: " + str(num_dues) + " -- UNKNWN_PCT: " + str(num_unknwn)
		print "---------------------------------------------------------------------"
		masked_list.append(num_masked)
		due_list.append(num_dues)
		sdc_list.append(num_sdcs)
		pot_due_list.append(0)
		num_interval_list.append(num_filtered_faults)
	return app_list, masked_list, sdc_list, due_list, pot_due_list, num_interval_list

def main():
	db_file = sys.argv[1]
	global c
	conn = sqlite3.connect(db_file)
	c = conn.cursor()

	#threshold = int(sys.argv[2])
	getFilteredOutcomes(c)
main()
