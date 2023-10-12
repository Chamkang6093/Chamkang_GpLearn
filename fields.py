"""The functions used to create and support programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs.s It also contains helper methods for a user to define their
own custom functions.
"""

import sympy
import numpy as np


class _Field(object):

    def __init__(self, name, class_name, unit, freq=1):
        self.name = name
        self.class_name = class_name
        self.unit = unit
        self.freq = freq
        


class _Field_Space(object):

    def __init__(self, combine_prob=0.25, add_div_prob = [0.3, 0.7]):

        self.combine_prob = combine_prob
        self.add_div_prob = add_div_prob
        if self.combine_prob > 1 or self.combine_prob < 0:
            raise AttributeError("Please reset attribute 'combine_prob' - it should be between 0 and 1 !")

        self.fields_set = (
            # The following fields are from - 'Social Media Data for Equity' dataset
            # Z score of sentiment
            _Field(name=sympy.Symbol('snt_social_value'), class_name="Social Media Data for Equity", unit=sympy.Symbol('value')),
            # Normalized tweet volume
            _Field(name=sympy.Symbol('snt_social_volume'), class_name="Social Media Data for Equity", unit=sympy.Symbol('volume')),

            # The following field is from - 'Model Rating Data' dataset
            # Ratings from 1-5, Strong buy=1, Buy=2, Hold=3, Sell=4, Strong Sell=5
            _Field(name=sympy.Symbol('rating'), class_name="Model Rating Data", unit=sympy.Symbol('number')),
            
            # Note: 'Fundamental Scores' dataset coverage values are very low, thus do not use that.
            # The following fields are from - 'Sentiment Data for Equity' dataset
            # relative sentiment volume
            _Field(name=sympy.Symbol('scl12_buzz'), class_name="Sentiment Data for Equity", unit=sympy.Symbol('volume')),
            # negative relative sentiment volume, fill nan with 0
            _Field(name=sympy.Symbol('snt_buzz'), class_name="Sentiment Data for Equity", unit=sympy.Symbol('volume')),
            # sentiment
            _Field(name=sympy.Symbol('scl12_sentiment'), class_name="Sentiment Data for Equity", unit=sympy.Symbol('value')),
            # negative sentiment, fill nan with 0
            _Field(name=sympy.Symbol('snt_value'), class_name="Sentiment Data for Equity", unit=sympy.Symbol('value')),
            # instrument type index
            _Field(name=sympy.Symbol('scl12_alltype_typevec'), class_name="Sentiment Data for Equity", unit=sympy.Symbol('number')),
            # negative return of relative sentiment volume
            _Field(name=sympy.Symbol('snt_buzz_ret'), class_name="Sentiment Data for Equity", unit=sympy.Symbol('ratio')),

            # The following fields are from - 'Systematic Risk Metrics' dataset
            # Beta to SPY in 30 | 60 | 90 | 360 Days
            _Field(name=sympy.Symbol('beta_last_30_days_spy'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_1'), freq=1/3),
            _Field(name=sympy.Symbol('beta_last_60_days_spy'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_1'), freq=1/3),
            _Field(name=sympy.Symbol('beta_last_90_days_spy'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_1'), freq=1/3),
            # Correlation to SPY in 30 | 60 | 90 | 360 Days
            _Field(name=sympy.Symbol('correlation_last_30_days_spy'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_2'), freq=1/3),
            _Field(name=sympy.Symbol('correlation_last_60_days_spy'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_2'), freq=1/3),
            _Field(name=sympy.Symbol('correlation_last_90_days_spy'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_2'), freq=1/3),
            # Systematic Risk Last 30 | 60 | 90 | 360 Days
            _Field(name=sympy.Symbol('systematic_risk_last_30_days'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_3'), freq=1/3),
            _Field(name=sympy.Symbol('systematic_risk_last_60_days'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_3'), freq=1/3),
            _Field(name=sympy.Symbol('systematic_risk_last_90_days'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_3'), freq=1/3),
            # Unsystematic Risk Last 30 | 60 | 90 | 360 Days - Relative to SPY
            _Field(name=sympy.Symbol('unsystematic_risk_last_30_days'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_4'), freq=1/3),
            _Field(name=sympy.Symbol('unsystematic_risk_last_60_days'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_4'), freq=1/3),
            _Field(name=sympy.Symbol('unsystematic_risk_last_90_days'), class_name="Systematic Risk Metrics", unit=sympy.Symbol('value_4'), freq=1/3)
        )
                 
        # For building trees and randomly choosing fields
        self.statistics = {}
        self.field_set = []
        self.prob_set = []
        for field in self.fields_set:
            self.field_set.append(field)
            self.prob_set.append(field.freq)
            ind = (field.class_name, field.unit)
            self.statistics[ind] = self.statistics.get(ind, 0)
            self.statistics[ind] += 1
        # Normalize probabilities
        self.prob_set = list(np.array(self.prob_set) / np.array(self.prob_set).sum())


    def choose_field(self, random_state):
        choice = random_state.choice(["combine", "field"], p=[self.combine_prob, 1 - self.combine_prob])
        if choice == "combine":
            while True:
                raw_field = random_state.choice(self.field_set, p=self.prob_set)
                if self.statistics[(raw_field.class_name, raw_field.unit)] >= 2:
                    field_set = []
                    prob_set = []
                    for field in self.fields_set:
                        if (field.class_name, field.unit) == (raw_field.class_name, raw_field.unit):
                            field_set.append(field)
                            prob_set.append(field.freq)
                    prob_set = list(np.array(prob_set) / np.array(prob_set).sum())
                    field1, field2 = random_state.choice(field_set, 2, p=prob_set, replace=False)
                    choice = random_state.choice(["add", "div"], p=self.add_div_prob)
                    if choice == "add":
                        random_num = random_state.rand()
                        return sympy.Function("add")(field1.name * np.round(random_num, 5), field2.name * np.round(1 - random_num, 5))
                    else:
                        return sympy.Function("div")(field1.name, field2.name)

        else:
            field = random_state.choice(self.field_set, p=self.prob_set)
            return field.name



# The following is main program
field_space = _Field_Space()