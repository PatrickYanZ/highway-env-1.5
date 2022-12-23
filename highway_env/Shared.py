import pandas as pd

class SharedState:
    thz_bss = []
    rf_bss = []
    vehicles = []
    bs_performance_table = pd.DataFrame()
    bs_assignment_table = pd.DataFrame()

    def __init__(self):
        self.thz_bss = []
        self.rf_bss = []
        self.vehicles = []
        self.bs_performance_table = pd.DataFrame()
        self.bs_assignment_table = pd.DataFrame()

    def get_bs_performance_table(self):
        return self.bs_performance_table
