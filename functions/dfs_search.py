from collections import deque
from Problem import AthletePerformanceProblem
from Node import Node
import time
class DFSSearch:
    def __init__(self, problem):
        self.problem = problem
        self.expanded_nodes = 0
        self.max_stack_size = 0
        self.max_depth = problem.target_day  # Use problem's target day
        self.solution = None

    def search(self):
        """Returns standardized search result format"""
        start_time = time.time()  # Start timing
        start_node = Node(state=self.problem.initial_state, costless=True)
        stack = deque([start_node])
        explored = set()

        while stack:
            current_node = stack.pop()
            day, fatigue, risk, performance, _ = current_node.state

            if self.problem.is_goal(current_node.state):
                print(f"Goal found at day {day}")
                self.solution = current_node
                break

            rounded_state = self._round_state(current_node.state)
            if rounded_state in explored:
                continue

            explored.add(rounded_state)

            for action in self.problem.actions():
                new_state = self.problem.apply_action(current_node.state, action)
                if self.problem.is_valid(new_state):
                    child_node = Node(new_state, parent=current_node, action=action, costless=True)
                    if child_node.depth <= self.max_depth:
                        stack.append(child_node)

            self.expanded_nodes += 1
            self.max_stack_size = max(self.max_stack_size, len(stack))
            
        self.execution_time = time.time() - start_time  # Calculate total time
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
            'message': 'Solution found with DFS',
            'schedule': schedule,
            'finalState': {
                'day': self.solution.state[0],
                'performance': self.solution.state[3],
                'fatigue': self.solution.state[1],
                'risk': self.solution.state[2]
            },
            'stats': self._get_stats(),
            'metrics': self._calculate_metrics(schedule)
        }

    def _build_schedule(self):
        """Build complete day-by-day schedule with explicit fields"""
        schedule = []
        current = self.solution
        while current.parent:
            entry = {
                'day': current.state[0],
                'intensity': current.action[0],
                'duration': current.action[1],
                'performance': current.state[3],
                'fatigue': current.state[1],
                'risk': current.state[2]
            }
            schedule.append(entry)
            current = current.parent
        
        # Reverse to show from day 1 to target day
        schedule.reverse()
        
        # Add missing days if any (to match A* format)
        full_schedule = []
        expected_day = 1
        for entry in schedule:
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

    def _calculate_metrics(self, schedule):
        """Calculate training metrics matching A* format"""
        return {
            'total_days': len(schedule),
            'rest_days': sum(1 for e in schedule if e['intensity'] == 0),
            'high_intensity_days': sum(1 for e in schedule if e['intensity'] >= 0.7),
            'total_workload': sum(e['intensity'] * e['duration'] for e in schedule)
        }

    def _get_stats(self):
        return {
            'nodesExplored': self.expanded_nodes,
            'maxQueueSize': self.max_stack_size,
            'executionTime': self.execution_time # Add timing logic if needed
        }

    def _round_state(self, state):
        return (
            state[0],  # day
            round(state[1], 1),  # fatigue
            round(state[2], 1),   # risk
            round(state[3], 1)   # performance
        )

    def reconstruct_path(self, node):
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return path[::-1]