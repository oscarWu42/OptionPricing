import pandas as pd
import numpy as np
import math


class Pricer:
    def __init__(self, spot, step_size=1.0, path_size=1000, rate=0.06, vol=0.2):
        self.spot = spot
        self.step_size = step_size
        self.path_size = path_size
        self.rate = rate
        self.vol = vol

    def __call__(self, opt):
        steps = int(opt.ttm//self.step_size)
        last_time_period = opt.ttm - self.step_size*steps
        df = math.exp(-self.rate*self.step_size)
        df_last = math.exp(-self.rate*last_time_period)

        def time_pass(x, dt):
            return x * math.exp((self.rate - self.vol ** 2 / 2) * dt + self.vol * np.random.normal(0, math.sqrt(dt)))

        def option_exercise(spot):
            return max(opt.strike - spot, 0)

        def predicted_value(x, *coef):
            return x * (coef[2] * x + coef[1]) + coef[0]

        d = {'t0': [self.spot for _ in range(self.path_size)]}
        data = pd.DataFrame(d)

        for i in range(steps):
            data['t{}'.format(i + 1)] = data['t0']
            time = self.step_size if not (i == steps) else last_time_period
            data.iloc[:, i + 1] = data.iloc[:, i].apply(lambda x: time_pass(x, time))

        value = pd.DataFrame()
        value = pd.DataFrame.reindex_like(value, data)
        value.iloc[:, steps] = data.iloc[:, steps].apply(option_exercise)

        for i in reversed(range(2, steps + 1)):
            df = df if i != steps else df_last
            value.iloc[:, i - 1] = value.iloc[:, i].apply(lambda x: x * df)

            current = pd.concat([data['t{}'.format(i - 1)], value.iloc[:, i]], axis=1)  # Get two consecutive cols
            ITM = current[current.iloc[:, 0] < opt.strike]
            ITM.iloc[:, -1] = ITM.iloc[:, -1].apply(option_exercise)

            # Regression
            polynomial_coefficients = np.polyfit(ITM.iloc[:, 0], ITM.iloc[:, 1] * df, 2)

            # Calculate prediction value based on the regression results
            ITM['predicted_' + 't{}'.format(i - 1)] = ITM.iloc[:, 0].apply(predicted_value,
                                                                           args=tuple(polynomial_coefficients))
            ITM[f'exercise_return_t{i - 1}'] = ITM.iloc[:, 0].apply(option_exercise)
            ITM['exercise'] = ITM[f'predicted_t{i - 1}'] < ITM[f'exercise_return_t{i - 1}']
            value.iloc[list(ITM[ITM['exercise'] == True].index), i - 1] = ITM[ITM['exercise'] == True][
                f'exercise_return_t{i - 1}']

        price = value.iloc[:, 1].apply(lambda x: x * df).mean()
        return price



