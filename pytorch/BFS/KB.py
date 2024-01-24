class KB(object):
	def __init__(self):
		self.entities = {}

	def addRelation(self, entity1, relation, entity2):
		if entity1 in self.entities:
			self.entities[entity1].append(Path(relation, entity2))
		else:
			self.entities[entity1] = [Path(relation, entity2)]

	def getPathsFrom(self, entity):
		return self.entities[entity]

	def removePath(self, entity1, entity2):
		for idx, path in enumerate(self.entities[entity1]):
			if(path.connected_entity == entity2):
				del self.entities[entity1][idx]
				break
		for idx, path in enumerate(self.entities[entity2]):
			if(path.connected_entity == entity1):
				del self.entities[entity2][idx]
				break
	#  源代码
	def pickRandomIntermediatesBetween(self, entity1, entity2, num):
		"""
		pick random intermediate entities and return.
		Args:
			entity1: entity1
			entity2: entity2
			num: the number of intermediate entities
		"""
		# TO DO: COULD BE IMPROVED BY NARROWING THE RANGE OF RANDOM EACH TIME ITERATIVELY CHOOSE AN INTERMEDIATE
		# from sets import Set - set is built-in class in python 3
		import random

		res = set()
		if num > len(self.entities) - 2:
			raise ValueError('Number of Intermediates picked is larger than possible', 'num_entities: {}'.format(len(self.entities)), 'num_itermediates: {}'.format(num))
		for i in range(num):
			intermediate = random.choice(list(self.entities.keys()))
			while intermediate in res or intermediate == entity1 or intermediate == entity2:
				intermediate = random.choice(list(self.entities.keys()))
			res.add(intermediate)
		return list(res)

	# 更新：去除多余重复的路径节点
	# def pickRandomIntermediatesBetween(self, entity1, entity2, num):
	# 	"""
	# 	随机选择中间实体并返回。
	# 	参数：
	# 		entity1：实体1
	# 		entity2：实体2
	# 		num：中间实体的数量
	# 	"""
	# 	import random

	# 	res = set()
	# 	available_entities = list(self.entities.keys())  # 初始时所有实体都是可选的

	# 	if num > len(available_entities) - 2:
	# 		raise ValueError('可选择的实体数量不足，无法选择所需数量的中间实体', 'num_entities: {}'.format(len(available_entities)), 'num_itermediates: {}'.format(num))

	# 	for i in range(num):
	# 		if not available_entities:
	# 			raise ValueError('可选择的实体数量不足，无法选择所需数量的中间实体')

	# 		# 在每次迭代中，从可选的实体中随机选择一个
	# 		intermediate = random.choice(available_entities)
			
	# 		# 从可选实体中移除已选择的实体，避免重复选择
	# 		available_entities.remove(intermediate)

	# 		# 确保选择的实体不是entity1、entity2或已经选择的中间实体
	# 		while intermediate in res or intermediate == entity1 or intermediate == entity2:
	# 			if not available_entities:
	# 				raise ValueError('可选择的实体数量不足，无法选择所需数量的中间实体')

	# 			intermediate = random.choice(available_entities)
	# 			available_entities.remove(intermediate)

	# 		# 将选择的实体添加到结果集合
	# 		res.add(intermediate)

	# 	return list(res)





	def __str__(self):
		string = ""
		for entity in self.entities:
			string += entity + ','.join(str(x) for x in self.entities[entity])
			string += '\n'
		return string


class Path(object):
	def __init__(self, relation, connected_entity):
		self.relation = relation
		self.connected_entity = connected_entity

	def __str__(self):
		return "\t{}\t{}".format(self.relation, self.connected_entity)

	__repr__ = __str__
