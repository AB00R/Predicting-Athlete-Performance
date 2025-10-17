from Node import Node
import Problem
import heapq
import time
import functools

class AStarSearch:
    def __init__(self, problem):
        self.problem = problem
        self.apply_cache = {}
        self.heuristic_cache = {}
        self.expanded_nodes = 0
        self.max_queue_size = 0
        self.execution_time = 0
        self.solution_node = None
        self._patch_apply_action()

    def _patch_apply_action(self):
        original = self.problem.apply_action
        @functools.wraps(original)
        def cached_apply(state, action):
            key = (self.get_state_key(state), action)
            return self.apply_cache.setdefault(key, original(state, action))
        self.problem.apply_action = cached_apply

    def get_state_key(self, state):
        day, fatigue, risk, performance, _ = state
        return (day, round(fatigue, 2), round(risk, 2), round(performance, 1))

    def search(self):
        """Returns standardized result format for API"""
        start_time = time.time()
        try:
            start_node = Node(self.problem.initial_state)
            start_node.g = 0
            start_node.h = self.problem.heuristic(start_node.state)
            start_node.f = start_node.g + start_node.h

            open_heap = [(start_node.f, id(start_node), start_node)]
            open_dict = {self.get_state_key(start_node.state): start_node}
            closed = set()

            while open_heap:
                self.max_queue_size = max(self.max_queue_size, len(open_heap))
                _, _, node = heapq.heappop(open_heap)
                state_key = self.get_state_key(node.state)
                
                if self.problem.is_goal(node.state):
                    self.solution_node = node
                    break
                
                if state_key in closed:
                    continue
                
                closed.add(state_key)
                self.expanded_nodes += 1

                for action in self.problem.actions():
                    child_state = self.problem.apply_action(node.state, action)
                    if not self.problem.is_valid(child_state):
                        continue

                    child_key = self.get_state_key(child_state)
                    if child_key in closed:
                        continue

                    g_new = node.g + self.problem.cost(node.state, action)
                    existing = open_dict.get(child_key)
                    
                    if existing and g_new >= existing.g:
                        continue

                    child_node = Node(child_state, parent=node, action=action)
                    child_node.g = g_new
                    child_node.h = self.problem.heuristic(child_state)
                    child_node.f = child_node.g + child_node.h

                    heapq.heappush(open_heap, (child_node.f, id(child_node), child_node))
                    open_dict[child_key] = child_node

        finally:
            self.execution_time = time.time() - start_time

        return self._format_result()

    def _format_result(self):
        """Standardized response format"""
        if not self.solution_node:
            return {
                'success': False,
                'message': 'No solution found',
                'stats': self._get_stats()
            }

        path = self._build_schedule()
        final_state = self.solution_node.state
        
        return {
            'success': True,
            'message': 'Solution found with A*',
            'schedule': path,
            'finalState': {
                'day': final_state[0],
                'performance': final_state[3],
                'fatigue': final_state[1],
                'risk': final_state[2]
            },
            'stats': self._get_stats(),
        }

    def _build_schedule(self):
        """Reconstruct path with day-by-day details"""
        schedule = []
        node = self.solution_node
        while node.parent:
            schedule.append({
                'day': node.state[0],
                'intensity': node.action[0],
                'duration': node.action[1],
                'performance': node.state[3],
                'fatigue': node.state[1],
                'risk': node.state[2]
            })
            node = node.parent
        return list(reversed(schedule))


    def _get_stats(self):
        return {
            'nodesExplored': self.expanded_nodes,
            'maxQueueSize': self.max_queue_size,
            'executionTime': self.execution_time
        }