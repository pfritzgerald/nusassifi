
# coding: utf-8
# In[1]:

#get_ipython().magic(u'matplotlib notebook')

"""
 IMPORTS
"""
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.metrics import silhouette_score
from pylab import cm
import sys, os

import specific_params as sp

# In[5]:
def main():
# db_file="../multiple_injections/2ordered/1injections_bbs.db"
	global db_dir
	db_dir="./"
	db_file=db_dir+"allapps_10K_.db"
	conn = sqlite3.connect(db_file)
	global c
	c = conn.cursor()
	global debug, plot_clusters, fig_count
	plot_clusters = False
	debug = False
	app_list = (u'gaussian',),#c.execute('SELECT App FROM BBProfile GROUP BY App;').fetchall()
	print app_list
	app_list = [row[0] for row in app_list]
    #(u'hotspot',),#[(u'bfs',),(u'gaussian',),]#(u'sad',),#(u'bfs',),#
	fig_count=0
	for app in app_list:
		print "\n---------\n" + app + "\n-----------"
#		INJSimMatrix(app)
#		fig_count += 1	
#		BBVSimilarityMatrix(app)
		Clustering(app, 800)
#		UseSimpoint(app, 400)
#    profileMemAccesses()

# In[]
def getOutcomesByInterval(app, plot=True):
	results = []
	interval_size = c.execute('SELECT IntervalSize FROM BBVIntervalSizes WHERE App IS \'%s\';'
						   % (app)).fetchone()[0]
#	interval_size /= 2
#    print "interval size: " + str(interval_size)
	total_dyn_inst_count = c.execute('SELECT MAX(InstIntervalID) FROM BBProfile WHERE App IS \'%s\';'
								  %(app)).fetchone()[0] * interval_size

#	total_dyn_inst_count /= 2
	masked_list = []
#	pot_due_list = []
	due_list = []
	sdc_list = []
	current_dyn_inst_id = 0
	current_interval = 0
	while current_dyn_inst_id < total_dyn_inst_count:
#         print('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
#                                 'AND Description LIKE \'Masked%%\' AND App is \'%s\' AND '\
#                                 'AppDynInstId>%d AND AppDynInstId<%d;'
#                                 %(app, current_dyn_inst_id, (current_dyn_inst_id+interval_size)))
		results.append([])
		num_faults = c.execute('SELECT COUNT(Results.ID) FROM Results WHERE  '\
						 ' App is \'%s\' AND '\
						 'AppDynInstId>%d AND AppDynInstId<%d;'
						 %(app, current_dyn_inst_id, (current_dyn_inst_id+interval_size))).fetchone()[0]

		num_masked = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
						 'AND Description LIKE \'Masked%%\' AND App is \'%s\' AND '\
						 'AppDynInstId>%d AND AppDynInstId<%d;'
						 %(app, current_dyn_inst_id, (current_dyn_inst_id+interval_size))).fetchone()[0]
		if num_faults != 0:
			masked_list.append(float(num_masked)/num_faults)
			results[current_interval].append(float(num_masked)/num_faults)
		else:
			masked_list.append(float(num_masked))
			results[current_interval].append(float(num_masked))
		num_due = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
					  'AND Description LIKE \'%%DUE%%\' AND App is \'%s\' AND '\
					  'AppDynInstId>%d AND AppDynInstId<%d;'
					  %(app, current_dyn_inst_id, (current_dyn_inst_id+interval_size))).fetchone()[0]
		if num_faults != 0:
			due_list.append(float(num_due)/num_faults)
			results[current_interval].append(float(num_due)/num_faults)

		else:
			due_list.append(float(num_due))
			results[current_interval].append(float(num_due))

#		num_pot_due = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
#						  'AND Description LIKE \'Pot DUE%%\' AND App is \'%s\' AND '\
#						  'AppDynInstId>%d AND AppDynInstId<%d;'
#						  %(app, current_dyn_inst_id, (current_dyn_inst_id+interval_size))).fetchone()[0]
#		if num_faults != 0:
#			pot_due_list.append(float(num_pot_due)/num_faults)
#			results[current_interval].append(float(num_pot_due)/num_faults)            
#		else:
#			pot_due_list.append(float(num_pot_due))
#			results[current_interval].append(float(num_pot_due))


		num_sdc = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
					  'AND Description LIKE \'SDC%%\' AND App is \'%s\' AND '\
					  'AppDynInstId>%d AND AppDynInstId<%d;'
					  %(app, current_dyn_inst_id, (current_dyn_inst_id+interval_size))).fetchone()[0]
		if num_faults != 0:
			sdc_list.append(float(num_sdc)/num_faults)
			results[current_interval].append(float(num_sdc)/num_faults)
		else:
			sdc_list.append(float(num_sdc))
			results[current_interval].append(float(num_sdc))

#         print "interval " + str(current_interval) + " - num masked: " + str(num_masked)
		current_dyn_inst_id += interval_size
		current_interval += 1
		sys.stdout.write('\r')
		sys.stdout.write('Overall Progress (All Intervals): '\
				   + str(round((float(current_dyn_inst_id)/total_dyn_inst_count),2)*100)+ '%')
		sys.stdout.flush()
	if plot:
		"""
		MASKED
		"""
		y_offset = 0
		plt.bar(np.arange(len(masked_list)),masked_list, bottom=y_offset, color='k', label="Masked")
		plt.title(app, size=20)

		"""
		DUE
		"""
		y_offset = [ x for x in  masked_list]
		plt.bar(np.arange(len(due_list)),due_list, bottom=y_offset, color='gray', label="DUE")

		"""
		SDC
		"""
		y_offset = [ x + y for x,y in zip(y_offset, due_list)]
		plt.bar(np.arange(len(sdc_list)),sdc_list, bottom=y_offset, color='lightgray', label="SDC")

		"""
		PotDUE
		"""
#		y_offset = [ x + y for x,y in zip(y_offset, sdc_list)]
#		plt.bar(np.arange(len(pot_due_list)), pot_due_list, bottom=y_offset, color='purple', label="PotDUE")

#		plt.ylim(0,1.5)
#		plt.legend(handles[::-1], labels[::-1],)    
#		interval_mark = str(interval_size/1000000) + "M" if (interval_size/1000000) >= 1 else str(interval_size/1000) + "K" 
		plt.xlabel("Inst-Intervals", size=18)#(Size:" + str(interval_mark)+ " insts)
		plt.ylabel("Outcome Percentage", size=18)
		ax = plt.gca()
		type(ax)
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles[::-1], labels[::-1], loc='best')		
		yvals = ax.get_yticks()
		ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in yvals], size=18)
		ax.tick_params(axis='x', labelsize=18)
		print "\n" + str(yvals)

#		plt.savefig(db_dir+"figs/time_plots/"+app+"10K-2_time")
		plt.show()
		plt.close()
    
	return results, interval_size

# In[]
def INJSimMatrix(app):
	results, interval_size = getOutcomesByInterval(app, plot=True)

#	print results
	num_intervals = len(results)
	print "\n---------\nSimMatrix for Injection Results\n--------------"
#	print "FI - NUM INTERVALS: \n"
	similarity_matrix = np.zeros((num_intervals, num_intervals))
    # print bbv[0]
	for interval_1 in range(0,num_intervals):
		for interval_2 in range(interval_1,num_intervals):
			sum_mntn_dist = sum_manhattan_distances(results[interval_1], results[interval_2])
			similarity_matrix [interval_1, interval_2] = sum_mntn_dist

#    print "\n" + str(len(results))
#    interval_size = c.execute('SELECT IntervalSize from BBVIntervalSizes WHERE App is \'%s\';'
#                           %(app)).fetchone()[0]
	plt.figure(fig_count)
	plotNormData(similarity_matrix, app,interval_size, True)
# In[]
def sum_manhattan_distances(interval_a, interval_b):
    assert(len(interval_a)==len(interval_b))
    sum_m_dist = 0
    interval_length = len(interval_a)
    for i in range(0,interval_length):
        sum_m_dist += euclidean(interval_a[i],interval_b[i])
    return sum_m_dist
# In[]
def getNormalizedMatrix(similarity_matrix):
    normalized_matrix = similarity_matrix
    max_value = max(max(i) for i in similarity_matrix)
    for i in range(0, similarity_matrix.shape[0]):
        for j in range(0, similarity_matrix.shape[1]):
            normalized_matrix[i,j] /= max_value
    return normalized_matrix

# In[]
def plotNormData(norm_values, app, interval_size, show=True):
	"""Takes in  normalized values and plots 
	the data
	"""
	# Initialize lists for plt.scatter
	x, y, colors = [], [], []

	# Determines the height of the array for the graph's Y-Value
	yval = norm_values.shape[0]
    
	# The size of each point
	# Dividing by 4.5 usually provides enough granularity, however this should 
	# be adjusted if a different resolution requirement is needed
	SIZE = yval/4.5

	#Adds data to x, y, and colors lists
	for i in range(0, yval):
		for j in range(i, yval):
			x.append(j)
			y.append(i)
			colors.append(norm_values[i,j])    
#	interval_mark = str(interval_size/1000000) +\
#	"M" if (interval_size/1000000) >= 1 else str(interval_size/1000) + "K" 
	#Plots data with gray colormap and aligns both axes to 0
	plt.scatter(x, y, c = colors, cmap=cm.gray, s = SIZE)
	plt.title(app, size=20)
	plt.xlabel("Inst-Intervals", size=18)# (Size:" + str(interval_mark)+ " insts)
	plt.ylabel("Inst-Intervals", size=18)# (Size:" + str(interval_mark)+ " insts)
	plt.gca().tick_params(axis='both', labelsize=18)

	plt.xlim(0)
	plt.ylim(0)

	#Inverts y axis to show similarity accurately
	plt.gca().invert_yaxis()
#    plt.savefig(db_dir+"figs/"+app)
	if show == True:
		plt.show()
#     plt.close()

# In[2]:

def getbbv_old(bb_id_list, func_name_list,total_dyn_inst_count,interval_size):
    current_dyn_inst_id = 0
    current_interval = 0
    bb_count = len(zip(bb_id_list,func_name_list))
    bbv = []
    while current_dyn_inst_id < total_dyn_inst_count:
        bb_counter = 0
        bbv.append([])
        for (bb_id,func_name) in zip(bb_id_list,func_name_list):
            bb_counter+=1
#             print bb_id, func_name
#             print ('SELECT SUM(Weight), NumInsts FROM BBProfile WHERE BasicBlockId=%d AND '\
#                     'FuncName LIKE \'%%%s%%\' AND App is \'%s\' AND AppDynInstId>%d AND AppDynInstId<%d;'
#             %(bb_id, func_name, app, current_dyn_inst_id,(current_dyn_inst_id+interval)))
            bbv_element = c.execute('SELECT SUM(Weight), NumInsts FROM BBProfile WHERE BasicBlockId=%d '\
                                    'AND FuncName IS \'%s\' AND App is \'%s\' AND '\
                                    'AppDynInstId>%d AND AppDynInstId<%d;'
                                   %(bb_id, func_name, app, current_dyn_inst_id, 
                                     (current_dyn_inst_id+interval_size))).fetchone()
            sys.stdout.write('\r')
            sys.stdout.write('Overall Progress (All Intervals): '\
                             + str((float(current_dyn_inst_id)/total_dyn_inst_count)*100)+ '% - ')
            sys.stdout.write('All BasicBlocks in Interval: ' + str((float(bb_counter)/bb_count)*100) + '%')
            sys.stdout.flush()
            
#             print bbv_element[1]
            if bbv_element[0] is None:
                bbv_data = 0
            else:
                bbv_data = bbv_element[0] * bbv_element[1]
            bbv[current_interval].append(float(bbv_data))
        current_dyn_inst_id += interval_size
        current_interval += 1

# In[1]
def profileMemAccesses():
    app_list = c.execute('SELECT App FROM MemAccesses GROUP BY App;').fetchall()
    app_list = [row[0] for row in app_list]
    #(u'hotspot',),#(u'kmeans',u'bfs'),#(u'gaussian',),#
    fig_count=0
    for app in app_list:
        print app
        global_loads= []
        global_stores = []
        nonglobal_loads = []
        nonglobal_stores = []
        #Get the basic blocks for this app
        plt.figure(fig_count)
        global_loads = c.execute('SELECT GlobalLoads from MemAccesses where App is \'%s\' ORDER BY'\
                                     ' IntervalId;'
                                     %(app)).fetchall()
        global_stores = c.execute('SELECT GlobalStores from MemAccesses where App is \'%s\' ORDER BY'\
                                     ' IntervalId;'
                                     %(app)).fetchall()
        nonglobal_loads = c.execute('SELECT NonGlobalLoads from MemAccesses where App is \'%s\' ORDER BY'\
                                     ' IntervalId;'
                                     %(app)).fetchall()
        nonglobal_stores = c.execute('SELECT NonGlobalStores from MemAccesses where App is \'%s\' ORDER BY'\
                                     ' IntervalId;'
                                     %(app)).fetchall()
        interval_list = c.execute('SELECT IntervalId FROM MemAccesses WHERE App is \'%s\' ORDER BY IntervalId;'
                                  %(app)).fetchall()
    
        plt.plot(interval_list, global_loads, label="GlobalLoads")
        plt.plot(interval_list, global_stores, label="GlobalStores")
        plt.plot(interval_list, nonglobal_loads, label="NonGlobalLoads")
        plt.plot(interval_list, nonglobal_stores, label="NonGlobalStores")
        plt.title(app)
        plt.legend()
        plt.show()
        fig_count += 1

# In[3]:
def getBBV(app):
	bbv = []
    #Get the basic blocks for this app
	basic_block_list = c.execute('SELECT BasicBlockId,FuncName from BBProfile where App is \'%s\' group'\
							  ' by FuncName, BasicBlockId;'
							  %(app)).fetchall()
	bb_id_list = [row[0] for row in basic_block_list]
	func_name_list = [row[1] for row in basic_block_list]
	max_interval = c.execute('SELECT MAX(InstIntervalId) FROM BBProfile WHERE App is \'%s\';'
						  % (app)).fetchone()[0]
#    bb_count = len(zip(bb_id_list,func_name_list))
	
	for inst_interval in range(0,max_interval):        
		bb_counter = 0
		bbv.append([])
		for (bb_id,func_name) in zip(bb_id_list,func_name_list):
			bb_counter+=1
#             print bb_id, func_name, inst_interval
#             print 'SELECT Weight, NumInsts FROM BBProfile WHERE BasicBlockId=%d '\
#                                 'AND FuncName IS \'%s\' AND App is \'%s\' AND '\
#                                 'InstInterval=%d;' %(bb_id, func_name, app, inst_interval)
			sum_live_out_max_rreg_used = c.execute('SELECT SUM(LiveOutMaxRRegUsed) FROM BBProfile WHERE '\
						   'FuncName IS \'%s\' AND App is \'%s\' AND '\
						   'InstIntervalId=%d;'
						   %(func_name, app, inst_interval)).fetchone()[0]
			bbv_element = c.execute('SELECT BBNumExecs, BBNumInsts, LiveOutMaxRRegUsed FROM BBProfile WHERE BasicBlockId=%d '\
						   'AND FuncName IS \'%s\' AND App is \'%s\' AND '\
						   'InstIntervalId=%d;'
						   %(bb_id, func_name, app, inst_interval)).fetchone()
#			num_mem_insts = c.execute('SELECT SUM(PUPCs.ID) FROM PUPCs,BBProfile WHERE  '\
#							 'BBProfile.BasicBlockId=PUPCs.BBId AND FuncName=FnName AND '\
#							 'BBProfile.App=PUPCs.App AND IsMem=1 AND BasicBlockId=%d '\
#							 'AND FuncName IS \'%s\' AND BBProfile.App is \'%s\' AND '\
#							 'InstIntervalId=%d;'
#						   %(bb_id, func_name, app, inst_interval)).fetchone()[0]
			
			old_write_len = 0
			sys.stdout.write('\r' + (' ' * old_write_len))            
			old_write_len = sys.stdout.write('\rOverall Progress (All Intervals): ' + 
						str(round((float(inst_interval)/max_interval),2)*100)+ '%')
#             sys.stdout.flush()
#             sys.stdout.write('All BasicBlocks in Interval: ' + 
#                             str(round((float(bb_counter)/bb_count),2)*100) + '%')
			sys.stdout.flush()

#			print bbv_element
#			if (num_mem_insts is None):
#				num_mem_insts = 1
#				print "Caught none num_mem_insts"
			if (bbv_element is None):
				bbv_data = 0
			else:
				live_out_ratio = 0 if sum_live_out_max_rreg_used==0 else float(bbv_element[2])/sum_live_out_max_rreg_used
				bbv_data = bbv_element[0] * bbv_element[1]# * live_out_ratio
			bbv[inst_interval].append(float(bbv_data))
#		if np.sum(bbv[inst_interval]) != 0:
#			bbv[inst_interval] /= np.sum(bbv[inst_interval])

	return bbv


# In[4]:
def choose_interval(policy,cluster_num,cluster_labels, cluster_centers,bbv, appl="gaussian", interval_size=1000):
    """
    choose a interval that belongs to cluster <cluster_num>
    <cluster_labels> is an array of size <num_intervals> where 
    the interval number is the index and the value at the index 
    indicates the cluster the interval belongs to
    For example: cluster_labels[4]=5 means that interval 4 belongs to cluster 5
    """
#    if debug: print "CHOOSING Interval for CLUSTER " + str(cluster_num)
    if policy == 'first':
        for i in range(0,len(cluster_labels)):
            if cluster_labels[i] == cluster_num:
                return_interval_id = i
                break    

    #choose interval closest to the cluster center
    elif policy == 'center':
        min_dist_to_center = euclidean(bbv[0],cluster_centers)
        return_interval_id = -1
        for i in range(0, len(cluster_labels)):
            if cluster_labels[i] == cluster_num:
#                 print bbv[i] - cluster_centers[i]
#                 dist_to_center = np.sum(bbv[i])
#                 scipy.spatial.distance.cityblock(
                dist_to_center = euclidean(bbv[i],cluster_centers)#
#                if debug: print "\tInterval " + str(i) + " - dist to center: " + str(dist_to_center)
                if dist_to_center <= min_dist_to_center:
#                    if debug: print "\t\tNEW minimum distance"
                    min_dist_to_center=dist_to_center
                    return_interval_id = i

    #Choose interval with the most faults
    elif policy == 'most_faults':
        return_interval_id = -1
        # Initially choose interval 0
        max_num_of_faults = 0
        for i in range(0, len(cluster_labels)):
            if cluster_labels[i] == cluster_num:
                num_of_faults = c.execute('SELECT COUNT(Results.ID) FROM Results WHERE App is \'%s\' AND '\
                                       'AppDynInstId>%d AND AppDynInstId<%d;'
                                    %(app, (i*interval_size), ((i+1)*interval_size))).fetchone()[0]
                if num_of_faults >= max_num_of_faults:
                    max_num_of_faults=num_of_faults
                    return_interval_id = i

    #Choose interval with the most faults
    elif policy == 'most_insts':
        return_interval_id = -1
        # Initially choose interval 0
        max_num_of_insts = 0
        for i in range(0, len(cluster_labels)):
            if cluster_labels[i] == cluster_num:
                num_of_insts = c.execute('SELECT NumGPRInsts FROM BBVIntervalSizes WHERE App is \'%s\' AND '\
                                       'IntervalId==%d;'
                                    %(app, i)).fetchone()[0]
                if num_of_insts >= max_num_of_insts:
                    max_num_of_insts=num_of_insts
                    return_interval_id = i
                    
#    if debug: print "\tFor cluster: " + str(cluster_num) + ", elected interval : " + str(return_interval_id)
    return return_interval_id

# In[]
#from __future__ import print_function
def BBVSimilarityMatrix(app):
		bbv=getBBV(app)
	
#		print "\n"
#		for 
		num_intervals = len(bbv)
		similarity_matrix = np.zeros((num_intervals, num_intervals))
#		# print bbv[0]
		for interval_1 in range(0,num_intervals):
			for interval_2 in range(interval_1,num_intervals):
				sum_mntn_dist = sum_manhattan_distances(bbv[interval_1], bbv[interval_2])
				similarity_matrix [interval_1, interval_2] = sum_mntn_dist
				#         if bbv[interval_] != 0:
				#         bbv[interval_] /= np.sum(bbv[interval_])
                #     print "interval " + str(interval_1) + ":\n " + str(bbv[interval_1])
                #norm_sim_matrix = getNormalizedMatrix(similarity_matrix)
		print "\n------------\nBBV PROFILING\n------------------"
		interval_size = c.execute('SELECT IntervalSize from BBVIntervalSizes WHERE App is \'%s\';'
							%(app)).fetchone()[0]
		plt.figure(fig_count)
		plotNormData(similarity_matrix, app,interval_size, True)
#		fig_count+=1


# In[23]:
def Clustering(app, num_faults):
#	app='bfs'
	fig_count=0
	max_sil_coeff = 0.0
	best_intervals = []
	best_num_cluster = 0
	best_interval_freqs = []
	best_interval_num_insts = []
	interval_size = c.execute('SELECT IntervalSize from BBVIntervalSizes WHERE App is \'%s\';'
		   %(app)).fetchone()[0]
	print "CLUSTERING " + app
	bbv = getBBV(app)
#	bbv, interval_size = getOutcomesByInterval(app, False)
	for num_clusters in range(2,20):
		clusters = KMeans(n_clusters=num_clusters)            
		clusters.fit(bbv)

		if plot_clusters:
			plt.figure(fig_count)
			plt.scatter(range(len(clusters.labels_)), clusters.labels_, marker="|")
			plt.title(app + " - NUM CLUSTERS: " + str(num_clusters))            
			fig_count += 1
			plt.figure(fig_count)
			plt.axis([0,num_clusters, 0, 150])
			plt.title(app + " - NUM CLUSTERS: " + str(num_clusters))
			plt.xticks(np.arange(num_clusters))
			plt.show()
			#        print "PHASE frequencies: " + str(cluster_freq)

		n,bins = np.histogram(clusters.labels_, num_clusters)
		cluster_freq = 1.0 * n/np.sum(n)
		intervals = []
		interval_num_insts = []

		for cluster in range(0,num_clusters):
			interval = choose_interval("center", cluster, clusters.labels_,
						  clusters.cluster_centers_[cluster], bbv, appl=app,
						  interval_size=interval_size)
			intervals.append(interval)
			num_insts = c.execute('SELECT NumGPRInsts FROM BBVIntervalSizes WHERE App is \'%s\' AND '\
							 'IntervalId==%d;'
							 %(app, interval)).fetchone()[0]
			interval_num_insts.append(num_insts)

		sil_coeff = silhouette_score(bbv, clusters.labels_, metric='euclidean')
		if sil_coeff > max_sil_coeff:
			max_sil_coeff = sil_coeff
			best_intervals = intervals[:]
			best_interval_freqs = cluster_freq[:]
			best_interval_num_insts = interval_num_insts[:]
			best_num_cluster = num_clusters

		if debug:
			print "\nNUM CLUSTERS=" + str(num_clusters)
			print "----------"
			print "INERTIA : %f and silhouette coefficient %f" % (clusters.inertia_, sil_coeff)
			print "----------\n"

        #         print "labels: " + str(clusters.labels_)
#        print "\nNUM CLUSTERS: %d" %(best_num_cluster)
	if debug:
		dumpDbgInfo(best_intervals, interval_size, best_interval_freqs, app)
	interval_fname = sp.app_dir[app] + "/interval.txt"
	if not debug: f = open(interval_fname, "w")
	interval_str = "%d\nApp:IntervalSize:NumFaultsPerInterval:IntervalList...\n"\
		%(interval_size)
	interval_str += "%s:%d:%d:%s\n" % (app, interval_size, num_faults, ':'.join(map(str, best_intervals)))
	print "\n%d" % interval_size
	print "App:IntervalSize:NumFaultsPerInterval:IntervalList..."
#        print "intervals chosen:"
#        print best_cluster_intervals
#        print interval_freqs
	print "%s:%d:%d:%s" % (app, interval_size, num_faults,':'.join(map(str, best_intervals)))
	for i in range(0,len(best_intervals)):
		print "%d:%d:%f"\
			% (best_intervals[i], best_interval_num_insts[i], best_interval_freqs[i])
		interval_str += "%d:%d:%f\n"\
			% (best_intervals[i], best_interval_num_insts[i], best_interval_freqs[i])

	if not debug:
		f.write(interval_str)
		f.close()
## 
# In[]
def dumpDbgInfo(best_intervals, interval_size, cluster_freq, app):

	adjusted_masked_rate = 0
	adjusted_sdc_rate = 0
	adjusted_due_rate = 0
	adjusted_pot_due_rate = 0
	error_list = []
	total_num_faults = []
	for (interval, cluster) in zip(best_intervals, np.arange(len(cluster_freq))):
		num_faults = c.execute('SELECT COUNT(Results.ID) FROM Results WHERE App is \'%s\' AND '\
					   'AppDynInstId>%d AND AppDynInstId<%d;'
					   %(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
		total_num_faults.append(num_faults)
		if num_faults < 50:
			print "-------------------\n< 50 FAULTS IN ONE OF THE CHOSEN INTERVALS\n-----------------------"
#				print "\t\tCluster Freg. " + str(cluster_freq[cluster]) +\
#					" NUMofFaults for Interval " + str(interval) + ": " + str(num_faults)
		num_masked = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
				   'AND Description LIKE \'Masked%%\' AND App is \'%s\' AND '\
				   'AppDynInstId>%d AND AppDynInstId<%d;'
				   %(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
		adjusted_masked_rate += (float(num_masked)/num_faults)*cluster_freq[cluster]
		num_sdc =c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
				   'AND Description LIKE \'SDC%%\' AND App is \'%s\' AND '\
				   'AppDynInstId>%d AND AppDynInstId<%d;'
				   %(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
		adjusted_sdc_rate += (float(num_sdc)/num_faults)*cluster_freq[cluster]
		num_due = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
				'AND Description LIKE \'DUE%%\' AND App is \'%s\' AND '\
				'AppDynInstId>%d AND AppDynInstId<%d;'
				%(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
		adjusted_due_rate += (float(num_due)/num_faults)*cluster_freq[cluster]
		num_pot_due = c.execute('SELECT COUNT(Results.ID) FROM Results, OutcomeMap WHERE OutcomeId=OutcomeMap.Id '\
					'AND Description LIKE \'Pot DUE%%\' AND App is \'%s\' AND '\
					'AppDynInstId>%d AND AppDynInstId<%d;'
					%(app, (interval*interval_size), ((interval+1)*interval_size))).fetchone()[0]
		adjusted_pot_due_rate += (float(num_pot_due)/num_faults)*cluster_freq[cluster]

	masked_rate = c.execute('SELECT 1.0*SUM(OutcomeMap.Description LIKE \'Masked:%%\')/COUNT(Results.ID) '\
				 'FROM Results,OutcomeMap WHERE OutcomeID==OutcomeMap.ID AND App is \'%s\';'
				 %(app)).fetchone()[0]
	masked_error = (adjusted_masked_rate - masked_rate)
	error_list.append(masked_error)
	sdc_rate = c.execute('SELECT 1.0*SUM(OutcomeMap.Description LIKE \'SDC:%%\')/COUNT(Results.ID) '\
				 'FROM Results,OutcomeMap WHERE OutcomeID==OutcomeMap.ID AND App is \'%s\';'
				 %(app)).fetchone()[0]
	sdc_error = (adjusted_sdc_rate - sdc_rate)
	error_list.append(sdc_error)
	due_rate = c.execute('SELECT 1.0*SUM(OutcomeMap.Description LIKE \'DUE:%%\')/COUNT(Results.ID) '\
				 'FROM Results,OutcomeMap WHERE OutcomeID==OutcomeMap.ID AND App is \'%s\';'
				 %(app)).fetchone()[0]
	due_error = (adjusted_due_rate - due_rate)
	error_list.append(due_error)
	potdue_rate = c.execute('SELECT 1.0*SUM(OutcomeMap.Description LIKE \'Pot DUE:%%\')/COUNT(Results.ID) '\
				 'FROM Results,OutcomeMap WHERE OutcomeID==OutcomeMap.ID AND App is \'%s\';'
				 %(app)).fetchone()[0]
	potdue_error = (adjusted_pot_due_rate - potdue_rate)
	error_list.append(potdue_error)
	print "\n====================================================================="
	print "Masked: " + str(adjusted_masked_rate) + " - SDC: " + str(adjusted_sdc_rate) + " - DUE: "\
		+ str(adjusted_due_rate) + " - Pot DUE: " + str(adjusted_pot_due_rate)
	print "----------------------------------------------------------------------"
	print "Masked +/- : " + str(masked_error)
	print "SDC +/- : " + str(sdc_error)
	print "DUE +/- : " + str(due_error)
	print "Pot DUE +/- : " + str(potdue_error)
	print "====================================================================="	
	print "Num Faults: " + ' '.join(str(e) for e in total_num_faults)
	plt.figure(fig_count)

	plt.bar(np.arange(4),error_list)
	plt.title(app+ " - " + str(len(best_intervals)) + " clusters" )
	plt.ylim(-0.25,0.25)
	plt.gca().grid()
	plt.xticks(np.arange(4), ['Masked', 'SDC', 'DUE', 'PotDUE'])
	plt.show()
# In[]
def UseSimpoint(app, num_faults):
	fipt_intervals = []
	num_intervals = 0
	fipt_intervals_weights = []
	fipt_intervals_num_insts = []
	interval_size = c.execute('SELECT IntervalSize from BBVIntervalSizes WHERE App is \'%s\';'
		   %(app)).fetchone()[0]
	print "CLUSTERING " + app
	bbv = getBBV(app)
	fv_file = open(app+".fv", "w")
	for interval in range(len(bbv)):
		fv_file.write("\nT")
		for bb_num in range(len(bbv[interval])):
			if bbv[interval][bb_num] != 0:
				fv_file.write(":" + str(bb_num+1) + ":" +	str(int(bbv[interval][bb_num])) + " ")
	fv_file.close()
	simpt_filename = "simpt.out"
	weight_filename = "weights.out"
	os.system('./simpoint -loadFVFile ' + app + '.fv -k 2:10 -dim \"noProject\" -saveSimpoints ' + simpt_filename + ' -saveSimpointWeights ' + weight_filename )
	with open("simpt.out") as f:
		fipts = f.readlines()
	fipt_intervals = [int(x.split(' ')[0]) for x in fipts]
	print "intervals chosen:"	
	print fipt_intervals
	with open("weights.out") as w:
		wts = w.readlines()
	fipt_intervals_weights = [float(x.split(' ')[0]) for x in wts]
	print fipt_intervals_weights
	
	print "SP APP DIR: " + app
	interval_fname = sp.app_dir[app] + "/interval.txt"
	f = open(interval_fname, "w")
	interval_str = "%d\nApp:IntervalSize:NumFaultsPerInterval:IntervalList...\n"\
		%(interval_size)
	interval_str += "%s:%d:%d:%s\n" % (app, interval_size, num_faults, ':'.join(map(str, fipt_intervals)))
	print "\n%d" % interval_size
	print "App:IntervalSize:NumFaultsPerInterval:IntervalList..."
#
	print "%s:%d:%d:%s" % (app, interval_size, num_faults,':'.join(map(str, fipt_intervals)))
	for interval in range(0,len(fipt_intervals)):
		num_insts  = c.execute('SELECT NumGPRInsts FROM BBVIntervalSizes WHERE App is \'%s\' AND '\
							 'IntervalId==%d;'
							 %(app, fipt_intervals[interval])).fetchone()[0]
		fipt_intervals_num_insts.append(num_insts)
		print "%d:%d:%f"\
			% (fipt_intervals[interval], fipt_intervals_num_insts[interval], fipt_intervals_weights[interval])
		interval_str += "%d:%d:%f\n"\
			% (fipt_intervals[interval], fipt_intervals_num_insts[interval], fipt_intervals_weights[interval])
#
	f.write(interval_str)
	f.close()

# In[]
main()
