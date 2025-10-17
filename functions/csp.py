import time
from Problem import AthletePerformanceProblem

class AthleteTrainingCSP:
    """
    Implementation of a Constraint Satisfaction Problem approach for athlete training planning
    using backtracking search.
    
    The CSP approach models the athlete training planning problem with:
    - Variables: Training activities for each day
    - Domains: Possible training intensities and durations for each day
    - Constraints: Fatigue limits and injury risk thresholds
    - Objective: Maximize performance
    """
    
    def __init__(self, 
        initial_state=(0, 0.0, 0.0, 1.0),
        target_day=30,
        target_fatigue=2.7,
        target_risk=0.5,
        sleep_duration = 7.5,
        sleep_quality = 3.0,
        stress_level = 2.5):
        
        # an object of the problem definition
        self.athlete_problem = AthletePerformanceProblem(
            initial_state=initial_state,
            sleep_duration=sleep_duration,
            sleep_quality=sleep_quality,
            stress_level=stress_level)
        
        # Thresholds
        self.target_day = target_day
        self.target_fatigue = target_fatigue
        self.target_risk = target_risk
        
        # Initialize state with history
        day, f, r, p = initial_state
        self.initial_state = (day, f, r, p, [
            {'load': 0.0,
            'fatigue': f,
            'injury_count': 0,
            'days_since_game': 0,
            'days_since_last_injury': 0}
        ])
        #physiological parameters
        self.sleep_quality = sleep_quality
        self.sleep_duration = sleep_duration
        self.stress_level = stress_level
        
        self.intensities = [0.0, 0.3, 0.6, 0.9]
        self.durations = [0, 60, 90, 120]
        
        # the domains: possible intensities and durations
        self.domains = set()
        
        # all possible pairs of intensity and duration
        for intensity in self.intensities:
            for duration in self.durations:
                if (intensity==0.0 and duration==0) or (intensity> 0.0 and duration>0):
                    self.domains.add((intensity, duration))

        self.transition_cache = {} # Cache for performance optimization
        
        # backtracking stats
        self.backtrack_stats = {
            'iterations': 0,
            'max_depth': 0,
            'pruning_count': 0
        }

    def is_goal(self, state):
        day, fatigue, risk, performance, _ = state
        return (day == self.target_day
                and fatigue <= self.target_fatigue
                and risk <= self.target_risk)
        
    def get_domains(self):
        return self.domains
    
    def apply_action(self, state, action):
        
        # caching to avoid recalculating
        cache_key = (str(state), str(action))
        if cache_key in self.transition_cache:
            return self.transition_cache[cache_key]
                
        day, fatigue, risk, performance, history = state
        
        basic_state = (day, fatigue, risk, performance, history) # tuple of only four elements !!!
        
        # apply the transition
        new_state = self.athlete_problem.apply_action(basic_state, action)
        
        self.transition_cache[cache_key] = new_state    # cache the answer
        return new_state
    
    def check_constraints(self, state):
        _, fatigue, risk, _, _ = state
        return fatigue <= self.target_fatigue and risk <= self.target_risk

    def evaluate_solution(self, solution):

        current_state = self.initial_state
        states = [current_state]
        
        # Apply each action in the solution
        for action in solution:
            current_state = self.apply_action(current_state, action)
            states.append(current_state)
        
        # Get final state
        final_day, final_fatigue, final_risk, final_performance, _ = states[-1]
        
        # Calculate metrics
        highest_fatigue = max(state[1] for state in states)
        highest_risk = max(state[2] for state in states)
        constraints_violated = any(state[1] > self.target_fatigue or state[2] > self.target_risk for state in states)
        
        # Calculate training metrics
        rest_days = sum(1 for action in solution if action[0] == 0.0 and action[1] == 0)
        high_intensity_days = sum(1 for action in solution if action[0] >= 0.7)
        total_workload = sum(self.athlete_problem.LOAD_PER_MIN.get(action[0], 0) * action[1] for action in solution)
        
        return {
            'final_performance': final_performance,
            'highest_fatigue': highest_fatigue,
            'final_fatigue': final_fatigue,
            'final_risk': final_risk,
            'highest_risk': highest_risk,
            'constraints_violated': constraints_violated,
            'days_trained': final_day,
            'rest_days': rest_days,
            'high_intensity_days': high_intensity_days,
            'total_workload': total_workload
        }

    def backtracking_search(self, time_limit=120):
        # reset our tracking stats for this search run
        self.backtrack_stats = {
            'iterations': 0,
            'max_depth': 0,
            'pruning_count': 0
        }
        
        solution_found = False 
        solution = []   
        
        # nested backtracking function that does the recursive search
        def _backtrack(assignment, current_state, depth=0):
            nonlocal solution_found, solution
            if solution_found:
                return True
            
            # track how many iterations we've done and how deep we've gone
            self.backtrack_stats['iterations'] += 1
            self.backtrack_stats['max_depth'] = max(self.backtrack_stats['max_depth'], depth)
            
            # make sure we haven't exceeded our time limit
            if time.time() - start_time > time_limit:
                return False
            
            day, fatigue, risk, performance, _ = current_state

            # check if we've reached our goal - the target day
            if day >= self.target_day:
                # done, we assgined each day with of intensity and duration
                solution_found = True
                solution = assignment.copy()
                return True
            
            # get possible actions for this day, ordered from most promising to least
            actions = self._get_ordered_domain_values(current_state)
            
            for action in actions:
                # try this action and see where it leads
                new_state = self.apply_action(current_state, action)
                
                # make sure this action doesn't break any constraints
                if self.check_constraints(new_state):
                    assignment.append(action)
                    
                    # recursively continue down this path
                    if _backtrack(assignment, new_state, depth + 1):
                        return True  
                    
                    # remove the last added assignment and redo the backtracking
                    assignment.pop()
                    
                    # if somehow a solution was found after backtracking, we can stop
                    if solution_found:
                        return True
            
            return False  # no solution found in this branch
        
        start_time = time.time()        
        _backtrack([], self.initial_state)
        end_time = time.time()
        
        if solution_found:
            print(f"First valid solution found in {end_time - start_time:.2f} seconds")
            # calculate how good this solution is
            evaluation = self.evaluate_solution(solution)
            print(f"Performance: {evaluation['final_performance']:.2f}")
        else:
            print(f"No solution found within time limit of {time_limit} seconds")
        
        return solution
    
    def search(self):
        start_time = time.time()
        solution = self.backtracking_search()
        execution_time = time.time() - start_time

        if not solution:
            return {
                'success': False,
                'message': 'No solution found',
                'stats': {
                    'nodesExplored': self.backtrack_stats['iterations'],
                    'maxQueueSize': self.backtrack_stats['max_depth'],
                    'executionTime': execution_time
                }
            }

        evaluation = self.evaluate_solution(solution)
        schedule = self._build_schedule(solution)
        
        return {
            'success': True,
            'message': 'Solution found with CSP',
            'schedule': schedule,
            'finalState': {
                'day': self.target_day,
                'performance': evaluation['final_performance'],
                'fatigue': evaluation['final_fatigue'],
                'risk': evaluation['final_risk']
            },
            'stats': {
                'nodesExplored': self.backtrack_stats['iterations'],
                'maxQueueSize': self.backtrack_stats['max_depth'],
                'executionTime': execution_time
            },
            'metrics': {
                'total_days': self.target_day,
                'rest_days': evaluation['rest_days'],
                'high_intensity_days': evaluation['high_intensity_days'],
                'total_workload': evaluation['total_workload'],
                'highest_fatigue': evaluation['highest_fatigue'],
                'highest_risk': evaluation['highest_risk']
            }
        }

    def _build_schedule(self, solution):
        """Create standardized schedule format"""
        schedule = []
        current_state = self.initial_state
        
        for action in solution:
            current_state = self.apply_action(current_state, action)
            day, fatigue, risk, performance, _ = current_state
            
            schedule.append({
                'day': day,
                'intensity': action[0],
                'duration': action[1],
                'performance': performance,
                'fatigue': fatigue,
                'risk': risk
            })
        
        # Fill in missing days if any
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
    #### FUNCTION DEFINITION: ORDERING DOMAIN VALUES
    # This function is the heart of the CSP optimizer's decision-making process.
    # It determines which training actions should be tried first during backtracking search.
    def _get_ordered_domain_values(self, state):
        """
        Advanced priority ordering of domain values (actions) with a sophisticated formula 
        that heavily prioritizes performance maximization above all else.
        """
        
        # Unpacking the state variables for easier access
        day, fatigue, risk, performance, history = state
        actions = self.get_domains()
        action_values = {}
        
        
        #### CONFIGURATION: OPTIMIZATION WEIGHTS
        PERFORMANCE_WEIGHT = 1000      # 1000 because its our primary objective
        
        # Headroom refers to the remaining capacity between the current fatigue/risk levels and their maximum allowable limits.
        # It indicates how much more stress the athlete can safely handle before reaching their physiological limits.
        # Higher headroom means more flexibility for intense training in the future.
        FATIGUE_HEADROOM_WEIGHT = 15   
        RISK_HEADROOM_WEIGHT = 20      
        
        ## Training pattern weights ensure physiologically sound training progression
        RECOVERY_BONUS = 30            # Value the recovery when fatigue is high
        LONG_TERM_POTENTIAL = 80      # Value actions with potential future payoff
        
        EFFICIENCY= 200
        
        #### PREPARATION: CALCULATE CURRENT STATE METRICS
        ## headroom: how much more fatigue/risk the athlete can handle
        # 0 means he can handle nothing
        fatigue_headroom = max(0, self.target_fatigue - fatigue)
        risk_headroom = max(0, self.target_risk - risk)
        
        # Count consecutive training days (days without rest)
        days_since_rest = sum(1 for h in reversed(history) if h.get('load', 0) > 0.1)
        
        # fatigue and risk status
        # >80% of maximum?  
        high_fatigue = fatigue > (self.target_fatigue * 0.8)
        high_risk = risk > (self.target_risk * 0.8)
        remaining_days_factor= ((day+1)/self.target_day)
        #### EVALUATION: evaluate EACH POSSIBLE ACTION
        for action in actions:
            intensity, duration = action
            
            future_state = self.apply_action(state, action)
            _, future_fatigue, future_risk, future_performance, _ = future_state
            
            ## violate constraints are given negative infinity (so it goes to the end when the domain is sorted)
            if future_fatigue > self.target_fatigue or future_risk > self.target_risk:
                action_values[action] = -float('inf') 
                self.backtrack_stats['pruning_count'] += 1 
                continue  
            
            perf_improvement = future_performance - performance
            
            ## Training efficiency metrics
            # Calculate standardized training load (hours equivalent)
            training_load = intensity * (duration / 60) 

            # Calculate performance gain per unit of training load (with safeguard against division by zero)
            efficiency = perf_improvement / max(0.5, training_load) if training_load > 0 else 0
            
            # Calculate future headroom values to see how action affects future training capacity
            future_fatigue_headroom = max(0, self.target_fatigue - future_fatigue)
            future_risk_headroom = max(0, self.target_risk - future_risk)
                        
            recovery_value = 0
            if intensity == 0 and duration == 0:  # Rest day
                if high_fatigue:
                    # we give bonuses for the rest if the athlete have high fatigue 
                    recovery_value = fatigue * RECOVERY_BONUS
                
                if days_since_rest > 3:  
                    # bonus if the athlete hasn't rested in >3 days
                    recovery_value += days_since_rest * 5
            
            
            # this one has the most important impact on finding an optimal schudule            
            # high intensity training -> future capacity
            # we then multiply by the factor of the remaining days because we want the performance to be maximized at the target day not before
            # this factor may be negative because remaining_days_factor
            long_term_value = (intensity)*(duration / 60)* (1-remaining_days_factor) 


            ## Performance valuation with exponential weighting
            # Square positive performance improvements to emphasize their value
            # Linear weighting for negative performance (avoids excessively punishing small negatives)
            performance_value = perf_improvement ** 2 * PERFORMANCE_WEIGHT if perf_improvement > 0 else perf_improvement * PERFORMANCE_WEIGHT
            
            #print(f"day:{day}")
            #print(f"long_term_value: {long_term_value}")
            #print(f"prformance: {performance_value}")
            #### FINAL SCORING: COMBINE ALL COMPONENTS
            action_values[action] = (
                performance_value*remaining_days_factor +
                efficiency * EFFICIENCY + 
                (future_fatigue_headroom - fatigue_headroom) * FATIGUE_HEADROOM_WEIGHT + 
                (future_risk_headroom - risk_headroom) * RISK_HEADROOM_WEIGHT +
                recovery_value + 
                long_term_value * LONG_TERM_POTENTIAL 
            )
        # Sort actions in descending order (best first) based on the scoring system we set
        return sorted(actions, key=lambda a: action_values.get(a, 0), reverse=True)