import queue
import time
from Node import Node
from Problem import AthletePerformanceProblem

class UCSSearch:
    def __init__(self, problem):
        self.problem = problem
        self.expanded_nodes = 0
        self.max_queue_size = 0
        self.execution_time = 0
        self.solution = None

    def search(self):
        max_depth= self.problem.target_day
        """Returns standardized search result format"""
        start_time = time.time()
        initial_node = Node(self.problem.initial_state, g=0)
        frontier = queue.PriorityQueue()
        frontier.put((0, initial_node))
        explored = {}
        best_cost = float('inf')

        while not frontier.empty():
            current_cost, current_node = frontier.get()
            
            if self.problem.is_goal(current_node.state):
                if current_cost < best_cost:
                    self.solution = current_node
                    best_cost = current_cost
                continue

            rounded_state = self._round_state(current_node.state)
            if rounded_state in explored and explored[rounded_state] <= current_cost:
                continue
                
            explored[rounded_state] = current_cost

            if current_node.depth >= max_depth:
                continue

            for action in self.problem.actions():
                new_state = self.problem.apply_action(current_node.state, action)
                if not self.problem.is_valid(new_state):
                    continue

                action_cost = self.problem.cost(current_node.state, action)
                new_cost = current_node.g + action_cost
                child_node = Node(new_state, parent=current_node, action=action, g=new_cost)
                frontier.put((new_cost, child_node))

            self.expanded_nodes += 1
            self.max_queue_size = max(self.max_queue_size, frontier.qsize())

        self.execution_time = time.time() - start_time
        return self._format_result()

    def _format_result(self):
        """Standardized result format for API"""
        if not self.solution:
            return {
                'success': False,
                'message': 'No solution found',
                'stats': self._get_stats()
            }

        schedule = self._build_schedule()
        
        return {
            'success': True,
            'message': 'Solution found with UCS',
            'schedule': schedule,
            'finalState': {
                'day': self.solution.state[0],
                'performance': self.solution.state[3],
                'fatigue': self.solution.state[1],
                'risk': self.solution.state[2]
            },
            'stats': self._get_stats(),
            'metrics': {
                'total_days': self.solution.state[0],
                'rest_days': sum(1 for entry in schedule if entry['intensity'] == 0),
                'high_intensity_days': sum(1 for entry in schedule if entry['intensity'] >= 0.7),
                'total_workload': sum(e['intensity'] * e['duration'] for e in schedule)
            }
        }

    def _build_schedule(self):
        """Build complete day-by-day schedule"""
        schedule = []
        current = self.solution
        while current.parent:
            schedule.append({
                'day': current.state[0],
                'intensity': current.action[0],
                'duration': current.action[1],
                'performance': current.state[3],
                'fatigue': current.state[1],
                'risk': current.state[2]
            })
            current = current.parent
        
        # Add missing days and reverse order
        full_schedule = []
        expected_day = 1
        for entry in reversed(schedule):
            while entry['day'] > expected_day:
                full_schedule.append(self._create_empty_day(expected_day))
                expected_day += 1
            full_schedule.append(entry)
            expected_day += 1
        
        return full_schedule

    def _create_empty_day(self, day):
        """Handle missing days in schedule"""
        return {
            'day': day,
            'intensity': 0.0,
            'duration': 0,
            'performance': 0,
            'fatigue': 0,
            'risk': 0
        }

    def _get_stats(self):
        return {
            'nodesExplored': self.expanded_nodes,
            'maxQueueSize': self.max_queue_size,
            'executionTime': self.execution_time
        }

    def _round_state(self, state):
        day, fatigue, risk, performance, _ = state
        return (
            day,
            round(fatigue, 2),
            round(risk, 2),
            round(performance, 2)
        )