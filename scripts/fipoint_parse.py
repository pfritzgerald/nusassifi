# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:13:20 2017

@author: previlon.f
"""
# In[1]:
import sqlite3

# In[2]"
def main():
    global db_dir
    db_dir= "./"
    db_file=db_dir + "data.db"
    conn = sqlite3.connect(db_file)
    global c
    c = conn.cursor()
    global debug
    debug = False
    app_list = c.execute('SELECT App FROM Results GROUP BY App;').fetchall()
    app_list = [row[0] for row in app_list]
    print "FI-PointResults\nApp,Masked,SDC,DUE,PotDUE"
    for app in app_list:
        #print app
        interval_size = c.execute('SELECT IntervalSize from BBVIntervalSizes '\
                                  'WHERE App is \'%s\';' %(app)).fetchone()[0]
        #print "Interval size: %d" % (interval_size)
        interval_list = c.execute('SELECT IntervalId,IntervalFrequency FROM '\
                                  'FIPointClusters WHERE App is \'%s\';' 
                                  % (app)).fetchall()
        interval_id_list = [row[0] for row in interval_list]
        frequency_list = [row[1] for row in interval_list]
        [adjusted_masked_rate, adjusted_sdc_rate, adjusted_due_rate, adjusted_pot_due_rate] = [0, 0, 0, 0]
        for (interval,freq) in zip(interval_id_list, frequency_list):
                    num_faults = c.execute('SELECT COUNT(Results.ID) FROM Results WHERE App is \'%s\' AND '\
                                           'AppDynInstId>%d AND AppDynInstId<%d;'
                                        %(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
		    if num_faults == 0:
			print "zero fault in interval %d " % (interval)
			continue
     #               print "\t\tCluster Freg. " + str(freq) +\
     #                   " NUMofFaults for Interval " + str(interval) + ": " + str(num_faults)
                    num_masked = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
                                           'AND Description LIKE \'Masked%%\' AND App is \'%s\' AND '\
                                           'AppDynInstId>%d AND AppDynInstId<%d;'
                                        %(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
                    adjusted_masked_rate += (float(num_masked)/num_faults)*freq
                    num_sdc =c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
                                       'AND Description LIKE \'SDC%%\' AND App is \'%s\' AND '\
                                       'AppDynInstId>%d AND AppDynInstId<%d;'
                                        %(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
                    adjusted_sdc_rate += (float(num_sdc)/num_faults)*freq
                    num_due = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
                                        'AND Description LIKE \'DUE%%\' AND App is \'%s\' AND '\
                                        'AppDynInstId>%d AND AppDynInstId<%d;'
                                        %(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
                    adjusted_due_rate += (float(num_due)/num_faults)*freq
                    num_pot_due = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
                                            'AND Description LIKE \'Pot DUE%%\' AND App is \'%s\' AND '\
                                            'AppDynInstId>%d AND AppDynInstId<%d;'
                                        %(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
                    adjusted_pot_due_rate+= (float(num_pot_due)/num_faults)*freq
       # print "====================================================================="
        print app + "," + str(adjusted_masked_rate) + "," + str(adjusted_sdc_rate) + ","\
                + str(adjusted_due_rate) + "," + str(adjusted_pot_due_rate)
            
# In[3]:
main()
