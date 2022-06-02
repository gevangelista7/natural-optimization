import itertools
from math import floor


class SearchHeuristics:

    def new_parameters(self):
        raise NotImplementedError


class ParameterGridSearch(SearchHeuristics):
    def __init__(self, T0_list: list, Tmin_list: list, alpha_list: list, max_iterations_list: list,
                 epsilon_list: list, delta_list: list):

        self.T0_list = T0_list[::-1]
        self.Tmin_list = Tmin_list[::-1]
        self.alpha_list = alpha_list[::-1]
        self.epsilon_list = epsilon_list[::-1]
        self.delta_list = delta_list[::-1]
        self.max_iterations_list = [floor(i) for i in max_iterations_list][::-1]

        self.combinations = []
        self.reset_combinations()
        self.last_reset = len(self.combinations)
        self.past_success = False

    def new_parameters(self):
        if self.past_success and self.last_reset < len(self.combinations):
            self.reset_combinations()
            self.past_success = False

        if len(self.combinations) > 0:
            res = self.combinations.pop()
        else:
            return None

        if len(res) < 6:
            return None
        else:
            return res

    def reset_combinations(self):
        self.last_reset = len(self.combinations)
        self.combinations = list(itertools.product(self.epsilon_list, self.delta_list, self.Tmin_list,
                                                   self.max_iterations_list, self.T0_list, self.alpha_list))

    def had_success(self):
        self.past_success = True

    def new_T0_basis(self, T0_min):
        self.T0_list = list(filter(lambda x: x >= T0_min, self.T0_list))

    def new_max_ite_basis(self, max_ite_min):
        self.max_iterations_list = list(filter(lambda x: x >= max_ite_min, self.max_iterations_list))



#
# class BadHeuristics(SearchHeuristics):
#     pass

    # if final_J > J_trigger_to_T:
    #     T0 *= T_adjust_factor
    #
    # if final_T/Tmin > final_T_ratio_adj_alpha:
    #     alpha *= alpha_adjust_factor
    #     T_condition_try_count += 1
    #     if T_condition_try_count == 2:
    #         delta *= delta_adjust_factor
    #         T_condition_try_count = 0
    #
    # if final_T / Tmin > final_T_ratio_adj_ite:
    #     max_iterations *= max_ite_adjust_factor
    #     alpha /= alpha_adjust_factor**2
    #
    # if max(history_D[-drop_D_length-i:-i])/min(history_D[-drop_D_length-i:-i]) > drop_D_criteria:
    #     max_iterations *= max_ite_adjust_factor

    #
    # J_trigger_level_completed = 3
    # J_trigger_to_T = 20
    # alpha_adjust_factor = 1.01
    # delta_adjust_factor = 1.02
    # T_adjust_factor = 1.5
    # final_T_ratio_adj_alpha = 1.05
    # final_T_ratio_adj_ite = 1.20
    # max_ite_adjust_factor = 1.2
    # drop_D_criteria = 1.2
    # drop_D_length = 0.2 * max_iterations
