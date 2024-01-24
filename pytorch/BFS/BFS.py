from queue import Queue
import random


def bfs(kb, entity1, entity2):
    # print("bfs:", entity1, entity2)
    res = FoundPaths(kb)

    res.mark_found(entity1, None, None)
    q = Queue()

    q.put(entity1)

    while not q.empty():
        cur_node = q.get()
        for path in kb.getPathsFrom(cur_node):
            next_entity = path.connected_entity
            connect_relation = path.relation
            if not res.is_found(next_entity):
                q.put(next_entity)
                res.mark_found(next_entity, cur_node, connect_relation)
            if next_entity == entity2:
                entity_list, path_list = res.reconstruct_path(entity1, entity2)
                return True, entity_list, path_list

    return False, None, None


class FoundPaths(object):
    def __init__(self, kb):
        self.entities = {}
        for entity, relations in kb.entities.items():
            self.entities[entity] = (False, "", "")

    def is_found(self, entity):
        return self.entities[entity][0]

    def mark_found(self, entity, prevNode, relation):
        self.entities[entity] = (True, prevNode, relation)

    def reconstruct_path(self, entity1, entity2):
        entity_list = []
        path_list = []
        cur_node = entity2
        while cur_node != entity1:
            entity_list.append(cur_node)

            path_list.append(self.entities[cur_node][2])
            cur_node = self.entities[cur_node][1]
        entity_list.append(cur_node)
        entity_list.reverse()
        path_list.reverse()
        return entity_list, path_list

    def __str__(self):
        res = ""
        for entity, status in self.entities.iteritems():
            res += entity + "[{},{},{}]".format(status[0], status[1], status[2])
        return res
