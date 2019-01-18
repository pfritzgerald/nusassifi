import sys

class KernelCfg:
	'''control flow graph for a kernel'''
	kname = ""
	graph = {}
	
	def __init__(self):
		self.kname = ""
		self.graph = {}
		self.entry = []
		self.exit = []

	def __str__(self):
		string = "\nKernel name: " + self.kname + "\n"
		for item in self.graph:
			string += "\t" + item + ": " + ", ".join(self.graph[item]) + "\n"
		return string

	# Add an edge to the graph
	def addEdge(self, origin, sink):
		if origin in self.graph:
			self.graph[origin].append(sink)
		else:
			self.graph[origin] = [sink]
		if sink not in self.graph:
			self.graph[sink] = []

	# Help with topological sorting (credit to www.geeksforgeeks.org/topological-sorting)
	def topologicalSortUtil(self, vertex, visited, stack):
		
		# Mark node as visited
		visited[vertex] = True

		# recursively visit all vertices adjacent to this vertex
		for v in self.graph[vertex]:
			if visited[v] == False:
				self.topologicalSortUtil(v, visited, stack)
		
		# Push current vertex to stack which has the results
		stack.insert(0, vertex)

	def topologicalSort(self):
		# First mark all vertices as not visited
		visited = dict([(i,False) for i in self.graph])
#		print "VISITED: " + str(visited)
		stack = []
		
		self.topologicalSortUtil(self.entry[0], visited, stack)
		# Call the recursive helper function to store the topological sort
		# starting from all vertices one by one
		for vertex in self.graph:
			if visited[vertex] == False:
				self.topologicalSortUtil(vertex, visited, stack)

		# return the topological sort
		#print "STACK returned for topo: " + str(stack)
		return stack
	
	def backEdgePassUtil(self, vertex, visited, stack):
		visited[vertex] = True
		stack[vertex] = True
		
		# recur for all neighbors
		# if any neighbor is visited, and in the stack then 
		# graph is cyclic
		for neighbor in self.graph[vertex]:
			if visited[neighbor] == False:
				self.backEdgePassUtil(neighbor, visited, stack)
			elif stack[neighbor] == True: # v->neighbor is a backedge
				#print "BACK EDGE: %s -> %s" % (vertex, neighbor)
				backedge_target = neighbor
				#self.graph[vertex].remove(neighbor) #eliminate back edge
				#self.addEdge(vertex,self.exit[0]) #add dummy edge v->exit
				#self.addEdge(self.entry[0],backedge_target) #add dummy edge entry->neighbor

		# vertex needs to be popped from recursion stack before ending function
			stack[vertex] = False
		#print "Rec STACK: " + str(stack)

	def backEdgePassOld(self):
		visited = dict([(i,False) for i in self.graph])
		stack = dict([(i,False) for i in self.graph])
		for vertex in self.topologicalSort():
			if visited[vertex] == False:
				self.backEdgePassUtil(vertex, visited, stack)

	def backEdgePass(self):
		"""
		Remove any back edge from the graph (BB to a previously executed BB)
		and point the origin of the back edge to an exit BB,
		then add an edge from an entry BB to the sink of the back edge 
		"""
		#visited = dict([(i,False) for i in self.graph])
		#stack = dict([(i,False) for i in self.graph])
		for vertex in self.graph:
			for neighbor in self.graph[vertex]:
				if int(neighbor.strip("BB")) < int(vertex.strip("BB")):
					#print "BACK EDGE: %s --> %s" % (vertex, neighbor)
					self.graph[vertex].remove(neighbor)
					self.addEdge(vertex, self.exit[0])
					self.addEdge(self.entry[0], neighbor)

def get_app_cfgs(cfg_file):
	app_cfgs = [] 
	cfg=open(cfg_file, 'r')

	for line in cfg:
		"""
		If this is the start of the profile info for a kernel
		"""
		if 'kernel' in line:
			app_cfgs.append(KernelCfg())
			kName = line.strip().split(",")[1]
			app_cfgs[-1].kname = kName
			#print "Added kernel " + kName + " to current cfg"

		"""
		If this is an entry Basic Block
		"""
		if 'entry' in line:
			entry_bb = line.strip().split(",")[1]
			app_cfgs[-1].entry.append(entry_bb)
			if entry_bb not in app_cfgs[-1].graph:
				app_cfgs[-1].graph[entry_bb] = []

			#print "Entry BB: " + str(entry_bb)

		"""
		If this is an exit basic block
		"""
		if 'exit' in line:
			exit_bb = line.strip().split(",")[1]
			app_cfgs[-1].exit.append(exit_bb)
			if exit_bb not in app_cfgs[-1].graph:
				app_cfgs[-1].graph[exit_bb] = []
			#print "Exit BB" + str(exit_bb)

		"""
		If this is an edge for the CFG
		"""
		if '->' in line:
			elements = line.split()
			origin = elements[0]
			sink = elements[2]
			app_cfgs[-1].addEdge(origin, sink)
			#print "added " + origin + " ==> " + sink + " to current cfg graph"

		"""
		If this is the end of parsing for this kernel
		"""
		if line.strip() == 'end':
			#print app_cfgs[-1]
			continue
	
	"""
	Clean up the entry and exit bbs, remove the BBs that are not in any path
	"""
	for kernel in app_cfgs:
		for bb in kernel.entry:
			if bb not in kernel.graph:
				kernel.entry.remove(bb)
		for bb in kernel.exit:
			if (bb not in kernel.graph):
				kernel.exit.remove(bb)
			elif  kernel.graph[bb]:
				kernel.exit.remove(bb)
	return app_cfgs

def path_sums(cfg, cfg_topo):
	"""
	Get the increment values from each node (vertex) to each neighbor
	The sum of increments from each node to another node is unique
	"""
	num_paths = dict([(i,0) for i in cfg.graph])
	edge_values = dict([(i, []) for i in cfg.graph])
	
	cfg_topo.reverse()
	for vertex in cfg_topo:
		if not cfg.graph[vertex]:
			num_paths[vertex] = 1
		else:
			num_paths[vertex] = 0
			for sink_vertex in cfg.graph[vertex]:
				edge_values[vertex].append(num_paths[vertex])
				num_paths[vertex] = num_paths[vertex] + num_paths[sink_vertex]
			assert(len(cfg.graph[vertex]) == len(edge_values[vertex]))
		#print "VERTEX: " + vertex + " - NUM PATHS: " + str(num_paths[vertex]) + " - Edges: " + str(edge_values[vertex])
	return num_paths, edge_values

#def print_path(path_id, cfg):

def main():
	cfg_file = sys.argv[1]
	cfg_list = get_app_cfgs(cfg_file)
	
	print str(len(cfg_list))
	for cfg in cfg_list:
		#print cfg
		cfg.backEdgePass()
		cfg_topo = cfg.topologicalSort()

		#print cfg_topo
		num_paths, edge_values = path_sums(cfg, cfg_topo)
		
		#print "--AFTER BACK EDGE PASS: "
		print cfg.kname
		#print cfg
		#print edge_values
		print str(sum([len(cfg.graph[i]) for i in cfg.graph]))
		for bb_from in cfg.graph:
			for (bb_to,inc) in zip(cfg.graph[bb_from],edge_values[bb_from]):
				print  bb_from.strip("BB") + " " + bb_to.strip("BB") + " " + str(inc)
		#print "NUM PATHS: " + str(num_paths)
		#print "EDGE Values: " + str(edge_values)
main()
