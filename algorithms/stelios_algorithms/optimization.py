import pandas as pd
import numpy as np
from copy import deepcopy
from scipy.optimize import linprog
from sklearn.preprocessing import MultiLabelBinarizer
import cvxpy as cp


class Optimization:

    def __init__(self, worker_dataframe, task_dataframe):
        self.worker_df = worker_dataframe
        self.task_df = task_dataframe

    def prepare_parameters(self, worker_df=None, task_df=None):
        """

        :param worker_df: header -> cost,skills,user_id,user_name
        :param task_df: header -> skill,skill_id,valuation
        :return:
        """

        if worker_df is None:
            worker_df = self.worker_df
        if task_df is None:
            task_df = self.task_df
        try:
            valuation = task_df['valuation']
            cost = worker_df['cost']
        except KeyError as e:
            raise KeyError("Keyerror") from e

        A = self.create_inequality_array(worker_df, task_df)
        b = np.zeros((task_df.shape[0],1))

        vector_valuation = np.array(valuation)
        vector_cost = np.array(cost) * -1
        c = np.concatenate((vector_valuation, vector_cost))

        bounds = [(0, 1) for _ in c]

        return c, A, b, bounds

    def create_inequality_array(self, worker_df=None, task_df=None):

        """
        # creates vector with 1s and 0s for every unique skill
        # --------------------------------------------------
        # genre
        #
        # 0[action, drama, fantasy]
        # 1[fantasy, action]
        # 2[drama]
        # 3[sci - fi, drama]
        # --------------------------------------------------
        # --------------------------------------------------
        # array([[1, 1, 1, 0],
        #        [1, 0, 1, 0],
        #        [0, 1, 0, 0],
        #        [0, 1, 0, 1]])
        # --------------------------------------------------
        :param worker_df:
        :param task_df:
        :return:
        """
        if worker_df is None:
            worker_df = self.worker_df
        if task_df is None:
            task_df = self.task_df

        mlb = MultiLabelBinarizer(classes=task_df.skill.values)
        worker_skills = worker_df['skills'].tolist()
        tasks_worker_array = mlb.fit_transform(worker_skills)
        eye_n = np.eye(len(task_df.skill.values))
        tasks_worker_array = tasks_worker_array.transpose()

        # n x n+m size
        # A represents the inequality xn< Sum(ym) for m: rn in Qm
        # identity matrix is the first part because x1 < sum..., x2<sum.. etc
        # so for the first row 1 followed by 24 zeros and then -1 for all the y's that can perform the task
        # multiply this row with the column vector X hwre 25 first elements are xn and 50 last elements are ym
        # and you get the inequality
        A = np.concatenate((eye_n, tasks_worker_array * -1), axis=1)

        return A

    def wd_lp_solver(self, shape, worker_df=None, task_df=None, c=None, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                     bounds=None, method='interior-point',
                     callback=None, options=None, x0=None):
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
        :param worker_df:
        :param task_df:
        :param c:
        :param shape:
        :param A_ub:
        :param b_ub:
        :param A_eq:
        :param b_eq:
        :param bounds:
        :param method:
        :param callback:
        :param options:
        :param x0:
        :return:
        """
        if worker_df is None:
            worker_df = self.worker_df
        if task_df is None:
            task_df = self.task_df

        if c is None:
            c, A_ub, b_ub, bounds = self.prepare_parameters(worker_df, task_df)

        opt = linprog(-1* c, A_ub, b_ub, A_eq, b_eq, bounds, method, callback, options, x0)
        x = opt.x[:shape]
        y = opt.x[shape:]

        task_df['x_star_index'] = x
        worker_df['y_star_index'] = y

        return opt, worker_df, task_df

    def wd_ip_solver(self, shape, worker_df, task_df, c=None, c_shape=None, A_ub=None, b_ub=None, verbose=1):
        """
        #TODO debug and documentation
        :param shape: len_x (or n or #tasks) shape is used only for the distiction of task vector and worker vector
                      if tasks are 150 then the first 150 elements are task variables and the rest are worker variables
        :param worker_df: pandas df.
        :param task_df: pandas df.
        :param c: vector.
        :param c_shape: n x 1
        :param A_ub:
        :param b_ub:
        :param verbose:
        :return:
        """

        if worker_df is None:
            worker_df = self.worker_df
        if task_df is None:
            task_df = self.task_df

        if c is None:
            c, A_ub, b_ub, bounds = self.prepare_parameters(worker_df, task_df)
            c_shape = c.shape
            b_ub = 0

        x = cp.Variable(c_shape, integer=True)
        c = c.transpose()
        constraints = [x >= 0, x <= 1, A_ub @ x <= b_ub]
        objective = cp.Maximize(x @ c)

        prob = cp.Problem(objective, constraints)
        prob.solve()
        if verbose:
            print("status:", prob.status)
            print("optimal value", prob.value)
            print("optimal var", x.value)
        xvals = x.value[:shape]
        yvals = x.value[shape:]

        task_df['x_star_index'] = xvals
        worker_df['y_star_index'] = yvals

        return prob, xvals, yvals, worker_df, task_df
