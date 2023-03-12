# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:05:07 2019

@author: Kiru Velswamy
"""

import numpy as np
import tensorflow as tf
import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import math
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.optimize import newton
import optuna

# global State_Dim
# global Action_Dim


def enum(**enums):
    return type('Enum', (), enums)


def signedSqr(x):
    if x == 0:
        return 0
    else:
        sign = x / abs(x)
        return sign * abs(x) ** 2


############################## DATA READING  ##############################

class PSVEnv(object):
    def __init__(self, sess):
        self.State_Dim = 5
        self.Action_Dim = 3

        # INITIAL STATE
        df_input = pd.read_excel(r'C:\Users\janse\OneDrive\Desktop\Compile.xlsx', sheet_name='PSC_FT_SS_FIX')

        i = 0
        u = np.zeros(13)
        u[0] = df_input._get_value(i, 'Qsl')
        u[1] = df_input._get_value(i, 'Qdil')
        u[2] = df_input._get_value(i, 'Qm')
        u[3] = df_input._get_value(i, 'Qt')
        u[4] = df_input._get_value(i, 'Qft')
        u[5] = df_input._get_value(i, 'Qfd')
        u[6] = df_input._get_value(i, 'a_bsl_1')
        u[7] = df_input._get_value(i, 'a_bsl_2')
        u[8] = df_input._get_value(i, 'a_bsl_3')
        u[9] = df_input._get_value(i, 'a_bsl_4')
        u[10] = df_input._get_value(i, 'a_ssl_1')
        u[11] = df_input._get_value(i, 'a_ssl_2')
        u[12] = df_input._get_value(i, 'a_ssl_3')

        # INITIAL STEADY-STATE
        x = np.zeros(39)
        x[0] = df_input._get_value(i, 'a_bf_1')
        x[1] = df_input._get_value(i, 'a_bf_2')
        x[2] = df_input._get_value(i, 'a_bf_3')
        x[3] = df_input._get_value(i, 'a_sf_1')

        x[4] = df_input._get_value(i, 'a_bmu_1')
        x[5] = df_input._get_value(i, 'a_bmu_2')
        x[6] = df_input._get_value(i, 'a_bmu_3')
        x[7] = df_input._get_value(i, 'a_smu_1')

        x[8] = df_input._get_value(i, 'a_bm_1')
        x[9] = df_input._get_value(i, 'a_bm_2')
        x[10] = df_input._get_value(i, 'a_bm_3')
        x[11] = df_input._get_value(i, 'a_bm_4')
        x[12] = df_input._get_value(i, 'a_sm_1')
        x[13] = df_input._get_value(i, 'a_sm_2')
        x[14] = df_input._get_value(i, 'a_sm_3')

        x[15] = df_input._get_value(i, 'a_st_1')
        x[16] = df_input._get_value(i, 'a_st_2')
        x[17] = df_input._get_value(i, 'a_st_3')
        x[18] = df_input._get_value(i, 'a_bt_4')

        x[19] = df_input._get_value(i, 'a_bfd_1')
        x[20] = df_input._get_value(i, 'a_bfd_2')
        x[21] = df_input._get_value(i, 'a_bfd_3')
        x[22] = df_input._get_value(i, 'a_bfd_4')
        x[23] = df_input._get_value(i, 'a_sfd_1')
        x[24] = df_input._get_value(i, 'a_sfd_2')
        x[25] = df_input._get_value(i, 'a_sfd_3')
        x[26] = 1408  # froth volume

        x[27] = df_input._get_value(i, 'a_aft')
        x[28] = df_input._get_value(i, 'a_bft_1')
        x[29] = df_input._get_value(i, 'a_bft_2')
        x[30] = df_input._get_value(i, 'a_bft_3')
        x[31] = df_input._get_value(i, 'a_bft_4')
        x[32] = df_input._get_value(i, 'a_sft_1')
        x[33] = df_input._get_value(i, 'a_sft_2')
        x[34] = df_input._get_value(i, 'a_sft_3')
        x[35] = df_input._get_value(i, 'a_bff_1')
        x[36] = df_input._get_value(i, 'a_bff_2')
        x[37] = df_input._get_value(i, 'a_bff_3')
        x[38] = df_input._get_value(i, 'a_bff_4')


        # To Run Experiment
        self.NO_HOURS = 1000
        self.SECONDS_PER_HOUR = 15*60
        self.MAIN_SAMPLE_TIME = self.SECONDS_PER_HOUR  # Hour
        self.LOWER_LOOP_SAMPLE_TIME = 2  # seconds - 1 min dt rate
        self.SP_GAP = 400  # int(self.NO_HOURS/2)  #Hour                                       # ????????? important
        self.EXPT_LENGTH = int(self.NO_HOURS * self.SECONDS_PER_HOUR / self.MAIN_SAMPLE_TIME)  # maximum number of steps per episode
        self.SWITCH_SP = int(self.SP_GAP * self.SECONDS_PER_HOUR / self.MAIN_SAMPLE_TIME)      # ????????
        self.LAB_SAMPLE_TIME = 2
        self.LAB_SAMPLE_COUNT = self.LAB_SAMPLE_TIME * self.SECONDS_PER_HOUR / self.MAIN_SAMPLE_TIME  # Lab Sample 2 perhour

        # Data Management
        # VF - volume of froth
        # VF_SP - Froth Volume setpoint
        # RR - recovery rate
        # DEL_QM - Change in middlings flow rate
        # REWARD - Based on Deviation and Action
        # QM - Actual middlings flow rate

        ############################## DATA READING  ##############################

        self.Data_Offset = enum(VF=0, VF_SP=1, LAST_ACTOR_STATE=3, CRITIC_STATE_BEGIN=0, DEV_VF=2, LAST_CRITIC_STATE=3, DEL_QM=3, REWARD=4, LAST_NORM_STATE=5, QM=5, LAST_OFFSET=6)
        self.Qt_Default = u[3]
        # State Ranges
        # Range of States          rho_fd   rho_f   rho_m   rho_t    Q_fd   Vf
        self.Max_List   = np.array([2600,   2600,   0,      0,       5,     self.Qt_Default * 1.2])  # self.Qm_Default*1.3])
        self.Min_List   = np.array([1000,   1000,   -30,    -1,     -5,     self.Qt_Default * 0.8])  # self.Qm_Default*0.7])
        self.Mean_List  = np.subtract(self.Max_List, self.Min_List) / 2
        self.Std_List   = np.abs(self.Mean_List) / 2


        self.NormState = np.zeros(self.Data_Offset.LAST_OFFSET, dtype=np.float32)
        self.States = np.zeros(self.Data_Offset.LAST_OFFSET, dtype=np.float32)

        # FIXED PARAMETER FOR PSC & FT
        self.A = 706.5  # PSC area m2
        self.mu = 0.000325  # kg/m *s
        self.g = 9.81  # m/s
        self.Vmix = 10  # m^3
        self.Vcyl = 2816  # m^3
        self.Vcon = 4592  # m^3
        self.Vm = 2296

        self.Vcell = 800
        self.Aft = 163  # ft cell area m2
        self.cd = 0.47  # ft cell bubble drag force

        # OPTIMIZED PARAMETER FOR PSC (psc_ft8)
        self.rho_b = np.array([800, 750, 700, 1050])  # kg/m3
        self.rho_s = np.array([1700, 2750, 2750])  # kg/m3
        self.rho_w = 971.8  # kg/m3
        self.d_b = np.array([303, 455, 562, 27]) * 1e-6  # m
        self.d_s = np.array([4, 171, 471]) * 1e-6  # m
        self.c1 = 0.41
        self.c2 = 2.7
        self.c3 = 0.34

        # OPTIMIZED FT CELL (psc_ft8)
        self.ae_b1 = 0.16  # (1/s)
        self.ae_b2 = 0.16  # (1/s)
        self.ae_b3 = 0.16  # (1/s)
        self.ae_b4 = 0.16  # (1/s)
        self.da = 650 * 1e-6  # m
        self.Qa = 1.2  # m3/s
        self.Vft = 500  # m3

        # INPUT VARIABLE
        self.Qsl = u[0]
        self.Qdil = u[1]
        self.Qm = u[2]
        self.Qt = u[3]
        self.Qft = u[4]
        self.Qfd = u[5]
        self.a_bsl_1 = u[6]
        self.a_bsl_2 = u[7]
        self.a_bsl_3 = u[8]
        self.a_bsl_4 = u[9]
        self.a_ssl_1 = u[10]
        self.a_ssl_2 = u[11]
        self.a_ssl_3 = u[12]

        self.X_s = [x[0],  # a_bf_1
                    x[1],  # a_bf_2
                    x[2],  # a_bf_3
                    x[3],  # a_sf_1

                    x[4],  # a_bmu_1
                    x[5],  # a_bmu_2
                    x[6],  # a_bmu_3
                    x[7],  # a_smu_1

                    x[8],  # a_bm_1
                    x[9],  # a_bm_2
                    x[10],  # a_bm_3
                    x[11],  # a_bm_4
                    x[12],  # a_sm_1
                    x[13],  # a_sm_2
                    x[14],  # a_sm_3

                    x[15],  # a_st_1
                    x[16],  # a_st_2
                    x[17],  # a_st_3
                    x[18],  # a_bt_4

                    x[19],  # a_bfd_1
                    x[20],  # a_bfd_2
                    x[21],  # a_bfd_3
                    x[22],  # a_bfd_4
                    x[23],  # a_sfd_1
                    x[24],  # a_sfd_2
                    x[25],  # a_sfd_3
                    x[26],  # froth volume

                    x[27],  # a_aft
                    x[28],  # a_bft_1
                    x[29],  # a_bft_2
                    x[30],  # a_bft_3
                    x[31],  # a_bft_4
                    x[32],  # a_sft_1
                    x[33],  # a_sft_2
                    x[34],  # a_sft_3
                    x[35],  # a_bff_1
                    x[36],  # a_bff_2
                    x[37],  # a_bff_3
                    x[38]]  # a_bff_4

        self.a_bsl = np.array([u[6], u[7], u[8]], u[9])  # ';   #??????
        self.a_bsl_default = self.a_bsl  # ??????
        self.a_ssl = np.array([u[10], u[11], u[12]])  # ';
        # self.V             = 1*10**3#; % m3
        # self.V_t           = 0.2*10**3#; % m3
        # self.V_mix         = 10#; % m3


        self.Vcell = 800
        self.Aft = 163  # ft cell area m2
        self.cd = 0.47  # ft cell bubble drag force

        # self.SP_LIST_VF = np.array([200,240,220,200,180,160,180,200,220,210,200,190])
        self.SP_LIST_VF = np.array([1400, 1680, 1540, 1400, 1260, 1120, 1260, 1400, 1540, 1470, 1400, 1330])

        self.IP_RANGE = 1  # Max %Change in one sample time in Input Signal
        self.QRANGE = 0.0005

        # ???????????????????????
        # Controller Tuning
        self.DEL_S_NORM = 900
        self.DEL_U_NORM = 1
        self.Lamda_U = 0.25

        # Resettable Variables

        self.SP_IDX = 0
        self.SP_RANGE = 10
        self.lenCount = 0
        self.a = 0
        self.SP = 1400  # self.SP_LIST_VF[self.SP_IDX]


        Qf = self.Qsl + self.Qdil - self.Qft - self.Qt
        self.Deviation = np.zeros(1)
        self.X = self.X_s
        self.EXPT_RECORDS = np.zeros([self.EXPT_LENGTH + 1, self.Data_Offset.LAST_OFFSET])
        self.States[self.Data_Offset.VF] = self.X_s[26]
        self.States[self.Data_Offset.VF_SP] = self.SP
        self.States[self.Data_Offset.DEV_VF] = 0  # self.SP

        #         self.States[self.Data_Offset.RR] = np.sum(self.X_0[1:4])*self.Qf/(np.sum(self.a_bo)*self.Qfd)
        self.States[self.Data_Offset.DEL_QM] = 0
        self.States[self.Data_Offset.REWARD] = 0
        self.States[self.Data_Offset.QM] = self.Qt
        self.NormState = np.true_divide(np.subtract(self.States[:self.Data_Offset.LAST_NORM_STATE],
                                                    self.Min_List[:self.Data_Offset.LAST_NORM_STATE]),
                                        np.subtract(self.Max_List[:self.Data_Offset.LAST_NORM_STATE],
                                                    self.Min_List[:self.Data_Offset.LAST_NORM_STATE]))
        self.Default_States = self.States
        self.Default_Norm_States = self.NormState
        self.episode = 0
        self.PSV_Reset = False
        self.Prev_VF = self.X_s[26]
        self.RR = (np.sum(self.X_s[0:3]) * Qf) / (np.sum(self.a_bsl) * self.Qsl)
        self.Reward = 0
        self.spa = 0

    def Env_step(self, a_sp, a, d):
        if self.lenCount == 0:
            self.reset_PSV()

        self.lenCount += 1

        self.a_sp_act = 0
        self.SP += a_sp * self.SP_RANGE
        self.spa = a_sp
        if self.SP > 1680:  # CHANGE
            self.a_sp_act = (1680 - (self.SP - a_sp * self.SP_RANGE)) / self.SP_RANGE
            self.SP = 1680  # CHANGE
        elif self.SP < 1120:  # CHANGE
            self.a_sp_act = -(1120 - (self.SP - a_sp * self.SP_RANGE)) / self.SP_RANGE
            self.SP = 1120
        self.violation = False

        self.a = a

        self.Qt = self.Qt + (self.a) * self.QRANGE  # (-10e-6)*(self.X[0]-self.SP)

        Qf = self.Qsl + self.Qdil - self.Qft - self.Qt


        for countInnerLoop in range(0, int(self.MAIN_SAMPLE_TIME / self.LOWER_LOOP_SAMPLE_TIME)):
            self.X += self.PSV_Model()

        self.Prev_VF = self.States[self.Data_Offset.VF]
        self.States[self.Data_Offset.VF] = self.X[26]
        self.Deviation = self.SP - self.States[self.Data_Offset.VF]
        self.States[self.Data_Offset.VF_SP] = self.SP
        self.States[self.Data_Offset.DEL_QM] = -np.abs(self.a)
        self.States[self.Data_Offset.DEV_VF] = a_sp * self.SP_RANGE  # -np.abs(self.Deviation)

        if (self.violation == False):
            reward = -np.abs(self.Deviation)
        #
        self.States[self.Data_Offset.REWARD] = reward
        self.States[self.Data_Offset.QM] = self.Qt

        self.EXPT_RECORDS[self.lenCount, :] = self.States
        self.RR = (np.sum(self.X[0:3]) * Qf) / (np.sum(self.a_bsl) * self.Qsl)

        # self.Reward = 100*signedSqr(self.RR-0.8) -0.005*np.abs(a_sp*self.SP_RANGE)**2
        self.Reward = 100 * signedSqr(self.RR - 0.8) #- 1000 * self.X[18] #- 0.005 * np.abs(a_sp * self.SP_RANGE) ** 2

        #

        if (self.lenCount == self.EXPT_LENGTH):  # or (np.abs(self.Deviation) < 2):
            doneExpt = True
            self.SP_IDX = 0
            self.lenCount = 0
        #
        else:
            doneExpt = False
            self.NormState = np.true_divide(np.subtract(self.States[:self.Data_Offset.LAST_NORM_STATE],
                                                        self.Min_List[:self.Data_Offset.LAST_NORM_STATE]),
                                            np.subtract(self.Max_List[:self.Data_Offset.LAST_NORM_STATE],
                                                        self.Min_List[:self.Data_Offset.LAST_NORM_STATE]))
        return doneExpt


    def _get_obs(self):
        theta, thetadot = self.X
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def reset_PSV(self):
        df_input = pd.read_excel(r'C:\Users\janse\OneDrive\Desktop\Compile.xlsx', sheet_name='PSC_FT_SS_FIX')

        i = 0
        u = np.zeros(13)
        u[0] = df_input._get_value(i, 'Qsl')
        u[1] = df_input._get_value(i, 'Qdil')
        u[2] = df_input._get_value(i, 'Qm')
        u[3] = df_input._get_value(i, 'Qt')
        u[4] = df_input._get_value(i, 'Qft')
        u[5] = df_input._get_value(i, 'Qfd')
        u[6] = df_input._get_value(i, 'a_bsl_1')
        u[7] = df_input._get_value(i, 'a_bsl_2')
        u[8] = df_input._get_value(i, 'a_bsl_3')
        u[9] = df_input._get_value(i, 'a_bsl_4')
        u[10] = df_input._get_value(i, 'a_ssl_1')
        u[11] = df_input._get_value(i, 'a_ssl_2')
        u[12] = df_input._get_value(i, 'a_ssl_3')

        # INITIAL STEADY-STATE
        x = np.zeros(39)
        x[0] = df_input._get_value(i, 'a_bf_1')
        x[1] = df_input._get_value(i, 'a_bf_2')
        x[2] = df_input._get_value(i, 'a_bf_3')
        x[3] = df_input._get_value(i, 'a_sf_1')

        x[4] = df_input._get_value(i, 'a_bmu_1')
        x[5] = df_input._get_value(i, 'a_bmu_2')
        x[6] = df_input._get_value(i, 'a_bmu_3')
        x[7] = df_input._get_value(i, 'a_smu_1')

        x[8] = df_input._get_value(i, 'a_bm_1')
        x[9] = df_input._get_value(i, 'a_bm_2')
        x[10] = df_input._get_value(i, 'a_bm_3')
        x[11] = df_input._get_value(i, 'a_bm_4')
        x[12] = df_input._get_value(i, 'a_sm_1')
        x[13] = df_input._get_value(i, 'a_sm_2')
        x[14] = df_input._get_value(i, 'a_sm_3')

        x[15] = df_input._get_value(i, 'a_st_1')
        x[16] = df_input._get_value(i, 'a_st_2')
        x[17] = df_input._get_value(i, 'a_st_3')
        x[18] = df_input._get_value(i, 'a_bt_4')

        x[19] = df_input._get_value(i, 'a_bfd_1')
        x[20] = df_input._get_value(i, 'a_bfd_2')
        x[21] = df_input._get_value(i, 'a_bfd_3')
        x[22] = df_input._get_value(i, 'a_bfd_4')
        x[23] = df_input._get_value(i, 'a_sfd_1')
        x[24] = df_input._get_value(i, 'a_sfd_2')
        x[25] = df_input._get_value(i, 'a_sfd_3')
        x[26] = 1408  # froth volume

        x[27] = df_input._get_value(i, 'a_aft')
        x[28] = df_input._get_value(i, 'a_bft_1')
        x[29] = df_input._get_value(i, 'a_bft_2')
        x[30] = df_input._get_value(i, 'a_bft_3')
        x[31] = df_input._get_value(i, 'a_bft_4')
        x[32] = df_input._get_value(i, 'a_sft_1')
        x[33] = df_input._get_value(i, 'a_sft_2')
        x[34] = df_input._get_value(i, 'a_sft_3')
        x[35] = df_input._get_value(i, 'a_bff_1')
        x[36] = df_input._get_value(i, 'a_bff_2')
        x[37] = df_input._get_value(i, 'a_bff_3')
        x[38] = df_input._get_value(i, 'a_bff_4')

        self.a = 0
        self.SP = 1400  # self.SP_LIST_VF[self.SP_IDX]
        # self.Qfd = 0.0056 # m^3/sec
        # self.Qm  = 0.002729122973783 # m^3/sec
        # self.Qt  = 0.012031377944153 #m^3/sec
        # self.Qfl = 0.0100 #m^3/sec
        # self.Qsl = u[5]
        # self.Qm  = u[2]
        # self.Qt  = u[3]
        # self.Qfl = u[1]
        # self.Qsl = u[0]
        # self.Qft = u[4]

        self.Qsl = u[0]
        self.Qdil = u[1]
        self.Qm = u[2]
        self.Qt = u[3]
        self.Qft = u[4]
        self.Qfd = u[5]

        self.PSV_Reset = False
        self.a_bsl = np.array([u[6], u[7], u[8]], u[9])
        self.a_ssl = np.array([u[10], u[11], u[12]])
        Qf = self.Qsl + self.Qdil - self.Qft - self.Qt
        self.Deviation = np.zeros(1)
        self.X = self.X_s
        self.EXPT_RECORDS = np.zeros([self.EXPT_LENGTH + 1, self.Data_Offset.LAST_OFFSET])
        self.States[self.Data_Offset.VF] = self.X_s[26]
        self.States[self.Data_Offset.VF_SP] = self.SP
        #         self.States[self.Data_Offset.RR] = np.sum(self.X_0[1:4])*self.Qf/(np.sum(self.a_bo)*self.Qfd)
        self.States[self.Data_Offset.DEL_QM] = 0
        self.States[self.Data_Offset.REWARD] = 0
        self.States[self.Data_Offset.QM] = self.Qt
        self.NormState = np.true_divide(np.subtract(self.States[:self.Data_Offset.LAST_NORM_STATE],
                                                    self.Min_List[:self.Data_Offset.LAST_NORM_STATE]),
                                        np.subtract(self.Max_List[:self.Data_Offset.LAST_NORM_STATE],
                                                    self.Min_List[:self.Data_Offset.LAST_NORM_STATE]))
        self.Default_States = self.States
        self.Default_Norm_States = self.NormState
        self.Prev_VF = self.X_s[26]
        self.RR = (np.sum(self.X_s[0:3]) * Qf) / (np.sum(self.a_bsl) * self.Qsl)
        self.Reward = 0
        self.spa = 0

    def PSV_Model(self):
        # MODEL ASSUMPTION FOR PSC
        # a_sf_1 = 0
        a_sf_2 = 0
        a_sf_3 = 0
        a_bf_4 = 0

        a_smu_2 = 0
        a_smu_3 = 0
        a_bmu_4 = 0

        a_bt_1 = 0
        a_bt_2 = 0
        a_bt_3 = 0
        # a_bt_4  = 0

        # u_sf_1 = 0
        u_sf_2 = 0
        u_sf_3 = 0
        u_bf_4 = 0

        u_smu_2 = 0
        u_smu_3 = 0
        u_bmu_4 = 0

        u_bm_1 = 0
        u_bm_2 = 0
        u_bm_3 = 0
        u_bm_4 = 0
        u_sm_1 = 0
        u_sm_2 = 0
        u_sm_3 = 0

        u_bt_1 = 0
        u_bt_2 = 0
        u_bt_3 = 0
        # u_bt_4 = 0

        # v_sf_1 = 0
        v_sf_2 = 0
        v_sf_3 = 0
        v_bf_4 = 0

        v_smu_2 = 0
        v_smu_3 = 0
        v_bmu_4 = 0

        v_bm_1 = 0
        v_bm_2 = 0
        v_bm_3 = 0
        v_bm_4 = 0
        v_sm_1 = 0
        v_sm_2 = 0
        v_sm_3 = 0

        v_bt_1 = 0
        v_bt_2 = 0
        v_bt_3 = 0
        # v_bt_4 = 0

        # ASSUMPTION FOR FT CELL
        Vtf_s1 = 0
        Vtf_s2 = 0
        Vtf_s3 = 0

        a_sff_1 = 0
        a_sff_2 = 0
        a_sff_3 = 0

        # Vf = self.X[0]
        # a_bf = np.array(self.X[1:4])
        # a_sf = self.X[4:7]
        # a_bm = self.X[7:10]
        # a_sm = self.X[10:13]
        # a_bt = self.X[13:16]
        # a_st = self.X[16:19]
        # a_bfd = self.X[19:22]
        # a_sfd = self.X[22:25]
        # a_wf = 1-np.sum(a_bf) - np.sum(a_sf)
        # a_wm = 1-np.sum(a_bm) - np.sum(a_sm)
        # a_wt = 1-np.sum(a_bt) - np.sum(a_st)

        # STATE VARIABLE
        a_bf_1 = self.X[0]
        a_bf_2 = self.X[1]
        a_bf_3 = self.X[2]
        a_sf_1 = self.X[3]
        a_bmu_1 = self.X[4]
        a_bmu_2 = self.X[5]
        a_bmu_3 = self.X[6]
        a_smu_1 = self.X[7]
        a_bm_1 = self.X[8]
        a_bm_2 = self.X[9]
        a_bm_3 = self.X[10]
        a_bm_4 = self.X[11]
        a_sm_1 = self.X[12]
        a_sm_2 = self.X[13]
        a_sm_3 = self.X[14]
        a_st_1 = self.X[15]
        a_st_2 = self.X[16]
        a_st_3 = self.X[17]
        a_bt_4 = self.X[18]
        a_bfd_1 = self.X[19]
        a_bfd_2 = self.X[20]
        a_bfd_3 = self.X[21]
        a_bfd_4 = self.X[22]
        a_sfd_1 = self.X[23]
        a_sfd_2 = self.X[24]
        a_sfd_3 = self.X[25]
        Vf = self.X[26]
        a_aft = self.X[27]
        a_bft_1 = self.X[28]
        a_bft_2 = self.X[29]
        a_bft_3 = self.X[30]
        a_bft_4 = self.X[31]
        a_sft_1 = self.X[32]
        a_sft_2 = self.X[33]
        a_sft_3 = self.X[34]
        a_bff_1 = self.X[35]
        a_bff_2 = self.X[36]
        a_bff_3 = self.X[37]
        a_bff_4 = self.X[38]

        # if (a_wf < 0):
        #   a_wf = 0
        # if (a_wm < 0):
        #   a_wm = 0
        # if (a_wt < 0):
        #   a_wt = 0

        # FROTH OVERFLOW
        Qff = self.Qm - self.Qft
        Qf = self.Qsl + self.Qdil - self.Qft - self.Qt

        # PSV LAYERS' VOLUME
        Vmu = self.Vcyl - Vf
        Vt = self.Vcon - self.Vm

        # rho_f = self.rho_w*a_wf + self.rho_b[0]*a_bf[0]+ self.rho_b[1]*a_bf[1]+self.rho_b[2]*a_bf[2]+ self.rho_s[0]*a_sf[0] +self.rho_s[1]*a_sf[1]+self.rho_s[2]*a_sf[2]
        # rho_m = self.rho_w*a_wm + self.rho_b[0]*a_bm[0]+ self.rho_b[1]*a_bm[1]+self.rho_b[2]*a_bm[2]+ self.rho_s[0]*a_sm[0] +self.rho_s[1]*a_sm[1]+self.rho_s[2]*a_sm[2]
        # rho_t = self.rho_w*a_wt + self.rho_b[0]*a_bt[0]+ self.rho_b[1]*a_bt[1]+self.rho_b[2]*a_bt[2]+ self.rho_s[0]*a_st[0] +self.rho_s[1]*a_st[1]+self.rho_s[2]*a_st[2]
        # V_m = self.V-Vf-self.V_t

        ###############################################################################################
        ####################################### QUOTE:PSC MODEL #######################################
        # WATER BALANCE
        a_wf = 1 - ((a_bf_1 + a_bf_2 + a_bf_3) + a_sf_1)
        a_wmu = 1 - ((a_bmu_1 + a_bmu_2 + a_bmu_3) + a_smu_1)
        a_wm = 1 - ((a_bm_1 + a_bm_2 + a_bm_3) + (a_sm_1 + a_sm_2 + a_sm_3) + a_bm_4)
        a_wt = 1 - ((a_st_1 + a_st_2 + a_st_3) + a_bt_4)

        # MIXTURE DENSITY
        rho_f = self.rho_w * a_wf + self.rho_b[0] * a_bf_1 + self.rho_b[1] * a_bf_2 + self.rho_b[2] * a_bf_3 + \
                self.rho_s[0] * a_sf_1
        rho_mu = self.rho_w * a_wmu + self.rho_b[0] * a_bmu_1 + self.rho_b[1] * a_bmu_2 + self.rho_b[2] * a_bmu_3 + \
                 self.rho_s[0] * a_smu_1
        rho_m = self.rho_w * a_wm + self.rho_b[0] * a_bm_1 + self.rho_b[1] * a_bm_2 + self.rho_b[2] * a_bm_3 + \
                self.rho_s[0] * a_sm_1 + self.rho_s[1] * a_sm_2 + self.rho_s[2] * a_sm_3 + self.rho_b[3] * a_bm_4
        rho_t = self.rho_w * a_wt + self.rho_s[0] * a_st_1 + self.rho_s[1] * a_st_2 + self.rho_s[2] * a_st_3 + \
                self.rho_b[3] * a_bt_4


        # QUOTE: VELOCITY CALCULATION
        def f1(z):
            v_bf_1 = z[0]
            v_bf_2 = z[1]
            v_bf_3 = z[2]
            v_sf_1 = z[3]
            v_bmu_1 = z[4]
            v_bmu_2 = z[5]
            v_bmu_3 = z[6]
            v_smu_1 = z[7]
            v_bt_4 = z[8]
            v_st_1 = z[9]
            v_st_2 = z[10]
            v_st_3 = z[11]
            v_wf = z[12]
            v_wmu = z[13]
            v_wt = z[14]

            f = np.zeros(15)
            f[0] = -v_wf - (Qf + self.A * (
                        a_bmu_1 * v_bf_1 + a_bmu_2 * v_bf_2 + a_bmu_3 * v_bf_3 + a_smu_1 * v_sf_1 + a_smu_2 * v_sf_2 + a_smu_3 * v_sf_3)) / (
                               a_wmu * self.A)
            f[1] = -v_wmu - (Qf + self.A * (
                        a_bm_1 * v_bmu_1 + a_bm_2 * v_bmu_2 + a_bm_3 * v_bmu_3 + a_sm_1 * v_smu_1 + a_sm_2 * v_smu_2 + a_sm_3 * v_smu_3)) / (
                               a_wm * self.A)
            f[2] = -v_wt + (self.Qt - self.A * (
                        a_bm_1 * v_bt_1 + a_bm_2 * v_bt_2 + a_bm_3 * v_bt_3 + a_sm_1 * v_st_1 + a_sm_2 * v_st_2 + a_sm_3 * v_st_3 + a_bm_4 * v_bt_4)) / (
                               a_wm * self.A)
            # FROTH
            f[3] = v_bf_1 - (self.g * (self.d_b[0] ** 2) * a_wf ** self.c2 * (self.rho_b[0] - rho_f) / (
                        18 * self.mu) + self.c1 * min(0, v_wf))
            f[4] = v_bf_2 - (self.g * (self.d_b[1] ** 2) * a_wf ** self.c2 * (self.rho_b[1] - rho_f) / (
                        18 * self.mu) + self.c1 * min(0, v_wf))
            f[5] = v_bf_3 - (self.g * (self.d_b[2] ** 2) * a_wf ** self.c2 * (self.rho_b[2] - rho_f) / (
                        18 * self.mu) + self.c1 * min(0, v_wf))
            f[6] = v_sf_1 - (self.g * (self.d_s[0] ** 2) * a_wf ** self.c2 * (self.rho_s[0] - rho_f) / (
                        18 * self.mu) + self.c1 * min(0, v_wf))
            # MIDDLING UPPER
            f[7] = v_bmu_1 - (self.g * (self.d_b[0] ** 2) * a_wmu ** self.c2 * (self.rho_b[0] - rho_mu) / (
                        18 * self.mu) + self.c1 * min(0, v_wmu))
            f[8] = v_bmu_2 - (self.g * (self.d_b[1] ** 2) * a_wmu ** self.c2 * (self.rho_b[1] - rho_mu) / (
                        18 * self.mu) + self.c1 * min(0, v_wmu))
            f[9] = v_bmu_3 - (self.g * (self.d_b[2] ** 2) * a_wmu ** self.c2 * (self.rho_b[2] - rho_mu) / (
                        18 * self.mu) + self.c1 * min(0, v_wmu))
            f[10] = v_smu_1 - (self.g * (self.d_s[0] ** 2) * a_wmu ** self.c2 * (self.rho_s[0] - rho_mu) / (
                        18 * self.mu) + self.c1 * min(0, v_wmu))
            # TAILING
            f[11] = v_bt_4 - (self.g * (self.d_b[3] ** 2) * a_wt ** self.c2 * (self.rho_b[3] - rho_t) / (
                        18 * self.mu) + self.c3 * max(0, v_wt))
            f[12] = v_st_1 - (self.g * (self.d_s[0] ** 2) * a_wt ** self.c2 * (self.rho_s[0] - rho_t) / (
                        18 * self.mu) + self.c3 * max(0, v_wt))
            f[13] = v_st_2 - (self.g * (self.d_s[1] ** 2) * a_wt ** self.c2 * (self.rho_s[1] - rho_t) / (
                        18 * self.mu) + self.c3 * max(0, v_wt))
            f[14] = v_st_3 - (self.g * (self.d_s[2] ** 2) * a_wt ** self.c2 * (self.rho_s[2] - rho_t) / (
                        18 * self.mu) + self.c3 * max(0, v_wt))

            return f

        init = np.full((15, 1), 0.0001)
        roots = fsolve(f1, init)

        v_bf_1 = roots[0]
        v_bf_2 = roots[1]
        v_bf_3 = roots[2]
        v_sf_1 = roots[3]
        v_bmu_1 = roots[4]
        v_bmu_2 = roots[5]
        v_bmu_3 = roots[6]
        v_smu_1 = roots[7]
        v_bt_4 = roots[8]
        v_st_1 = roots[9]
        v_st_2 = roots[10]
        v_st_3 = roots[11]
        v_wf = roots[12]
        v_wmu = roots[13]
        v_wt = roots[14]

        # UNQUOTE: VELOCITY CALCULATION

        # VI = ((a_bm[0]*v_bm[0] +a_bm[1]*v_bm[1]+a_bm[2]*v_bm[2]) -(a_bf[0]*v_bf[0] + a_bf[1]*v_bf[1] + a_bf[2]*v_bf[2]))/(a_bm[0]+a_bm[1]+a_bm[2]-a_bf[0]-a_bf[1]-a_bf[2])

        # FROTH-MIDDLING FLUX
        def flux(VI, v_mu, a_f, a_mu):
            if VI > v_mu:
                flux = a_mu * self.A * (VI - v_mu)
            else:
                flux = a_f * self.A * (VI - v_mu)
            return flux

        # INTERFACE VELOCITY
        VI = (((a_bm_1 * v_bmu_1 + a_bm_2 * v_bmu_2 + a_bm_3 * v_bmu_3) - (
                    a_bmu_1 * v_bf_1 + a_bmu_2 * v_bf_2 + a_bmu_3 * v_bf_3)) / (
                          a_bmu_1 + a_bmu_2 + a_bmu_3 - a_bf_1 - a_bf_2 - a_bf_3))

        # F = np.zeros(25)
        #   F[0] = VI * self.A
        #
        #   for spCount in range(0,3):
        #     if (v_bm[spCount] < VI):
        #       F[1+spCount] = (1/Vf) * (-a_bm[spCount]*v_bm[spCount]*self.A + VI*self.A*(a_bm[spCount]-a_bf[spCount])-self.Qf*a_bf[spCount])
        #       F[7+spCount] = (1/V_m) * ((self.Qsl + self.Qfl) * a_bfd[spCount] - self.Qm * a_bm[spCount] - a_bm[spCount] * v_bt[spCount] * self.A + a_bm[spCount] * v_bm[spCount] * self.A)
        #     else:
        #       F[1+spCount] = (1/Vf) * (-a_bf[spCount]*v_bm[spCount]*self.A - self.Qf*a_bf[spCount])
        #       F[7+spCount] = (1/V_m) * ((self.Qsl + self.Qfl) * a_bfd[spCount] - self.Qm * a_bm[spCount] - a_bm[spCount] * v_bt[spCount] * self.A + a_bf[spCount] * v_bm[spCount] * self.A + VI * self.A * (a_bm[spCount] - a_bf[spCount]))
        #
        #     if (v_sm[spCount] < VI):
        #       F[4+spCount] = (1/Vf) * (-a_sm[spCount]*v_sm[spCount]*self.A + VI*self.A*(a_sm[spCount]-a_sf[spCount]) - self.Qf*a_sf[spCount])
        #       F[10+spCount] = (1/V_m)* ((self.Qsl + self.Qfl) * a_sfd[spCount] - self.Qm * a_sm[spCount] - a_sm[spCount] * v_st[spCount] * self.A + a_sm[spCount] * v_sm[spCount] * self.A)
        #     else:
        #       F[4+spCount] = (1/Vf) * (-a_sf[spCount]*v_sm[spCount]*self.A - self.Qf*a_sf[spCount])
        #       F[10+spCount] = (1/V_m) * ((self.Qsl + self.Qfl) * a_sfd[spCount] - self.Qm * a_sm[spCount] - a_sm[spCount] * v_st[spCount] * self.A + a_sf[spCount] * v_sm[spCount] * self.A + VI * self.A * (a_sm[spCount] - a_sf[spCount]))
        #
        #     F[13+spCount] = (1/self.V_t)*(a_bm[spCount]*v_bt[spCount]*self.A - self.Qt*a_bt[spCount])
        #     F[16+spCount] = (1/self.V_t)*(a_sm[spCount]*v_st[spCount]*self.A - self.Qt*a_st[spCount])
        #     F[19+spCount] = (1/self.V_mix)*(self.Qsl * (self.a_bsl[spCount] - a_bfd[spCount]) - self.Qfl * a_bfd[spCount])
        #     F[22+spCount] = (1/self.V_mix)*(self.Qsl * (self.a_ssl[spCount] - a_sfd[spCount]) - self.Qfl * a_sfd[spCount])

        ########## VOLUME FRACTION CHANGE ##########
        f = np.zeros(39)
        # FROTH
        f[0] = 1 / Vf * (flux(VI, v_bf_1, a_bf_1, a_bmu_1) - Qf * a_bf_1 - (a_bf_1 * self.A * VI))
        f[1] = 1 / Vf * (flux(VI, v_bf_2, a_bf_2, a_bmu_2) - Qf * a_bf_2 - (a_bf_2 * self.A * VI))
        f[2] = 1 / Vf * (flux(VI, v_bf_3, a_bf_3, a_bmu_3) - Qf * a_bf_3 - (a_bf_3 * self.A * VI))
        f[3] = 1 / Vf * (flux(VI, v_sf_1, a_sf_1, a_smu_1) - Qf * a_sf_1 - (a_sf_1 * self.A * VI))

        # MIDDLING UPPER
        f[4] = 1 / Vmu * (-flux(VI, v_bf_1, a_bf_1, a_bmu_1) - a_bm_1 * self.A * v_bmu_1 + a_bmu_1 * self.A * VI)
        f[5] = 1 / Vmu * (-flux(VI, v_bf_2, a_bf_2, a_bmu_2) - a_bm_2 * self.A * v_bmu_2 + a_bmu_2 * self.A * VI)
        f[6] = 1 / Vmu * (-flux(VI, v_bf_3, a_bf_3, a_bmu_3) - a_bm_3 * self.A * v_bmu_3 + a_bmu_3 * self.A * VI)
        f[7] = 1 / Vmu * (-flux(VI, v_sf_1, a_sf_1, a_smu_1) - a_sm_1 * self.A * v_smu_1 + a_smu_1 * self.A * VI)

        # MIDDLING (SOURCE ZONE)
        f[8] = 1 / self.Vm * (self.Qfd * a_bfd_1 + Qff * a_bff_1 - self.Qm * a_bm_1 + a_bm_1 * self.A * v_bmu_1)
        f[9] = 1 / self.Vm * (self.Qfd * a_bfd_2 + Qff * a_bff_2 - self.Qm * a_bm_2 + a_bm_2 * self.A * v_bmu_2)
        f[10] = 1 / self.Vm * (self.Qfd * a_bfd_3 + Qff * a_bff_3 - self.Qm * a_bm_3 + a_bm_3 * self.A * v_bmu_3)
        f[11] = 1 / self.Vm * (self.Qfd * a_bfd_4 + Qff * a_bff_4 - self.Qm * a_bm_4 - a_bm_4 * self.A * v_bt_4)
        f[12] = 1 / self.Vm * (
                self.Qfd * a_sfd_1 + - self.Qm * a_sm_1 - a_sm_1 * self.A * v_st_1 + a_sm_1 * self.A * v_smu_1)
        f[13] = 1 / self.Vm * (self.Qfd * a_sfd_2 + - self.Qm * a_sm_2 - a_sm_2 * self.A * v_st_2)
        f[14] = 1 / self.Vm * (self.Qfd * a_sfd_3 + - self.Qm * a_sm_3 - a_sm_3 * self.A * v_st_3)

        # TAILING
        f[15] = 1 / Vt * (a_sm_1 * self.A * v_st_1 - a_st_1 * self.Qt)
        f[16] = 1 / Vt * (a_sm_2 * self.A * v_st_2 - a_st_2 * self.Qt)
        f[17] = 1 / Vt * (a_sm_3 * self.A * v_st_3 - a_st_3 * self.Qt)
        f[18] = 1 / Vt * (a_bm_4 * self.A * v_bt_4 - a_bt_4 * self.Qt)
        ########## VOLUME FRACTION CHANGE ##########

        # MIXER
        f[19] = 1 / self.Vmix * (self.Qsl * (self.a_bsl_1 - a_bfd_1) - self.Qdil * a_bfd_1)
        f[20] = 1 / self.Vmix * (self.Qsl * (self.a_bsl_2 - a_bfd_2) - self.Qdil * a_bfd_2)
        f[21] = 1 / self.Vmix * (self.Qsl * (self.a_bsl_3 - a_bfd_3) - self.Qdil * a_bfd_3)
        f[22] = 1 / self.Vmix * (self.Qsl * (self.a_bsl_4 - a_bfd_4) - self.Qdil * a_bfd_4)
        f[23] = 1 / self.Vmix * (self.Qsl * (self.a_ssl_1 - a_sfd_1) - self.Qdil * a_sfd_1)
        f[24] = 1 / self.Vmix * (self.Qsl * (self.a_ssl_2 - a_sfd_2) - self.Qdil * a_sfd_2)
        f[25] = 1 / self.Vmix * (self.Qsl * (self.a_ssl_3 - a_sfd_3) - self.Qdil * a_sfd_3)

        f[26] = VI * self.A
        ###################################### UNQUOTE:PSC MODEL ######################################
        ###############################################################################################

        ###############################################################################################
        ##################################### QUOTE:FT cell MODEL #####################################
        vb = ((self.g * self.da) / (3 * self.cd)) ** 0.5  # bubble velocity
        Vtf_a = vb * self.Aft * a_aft
        Vtf_b1 = self.ae_b1 * 6 * self.d_b[0] * a_aft * a_bft_1 * self.Vft / self.da
        Vtf_b2 = self.ae_b2 * 6 * self.d_b[1] * a_aft * a_bft_2 * self.Vft / self.da
        Vtf_b3 = self.ae_b3 * 6 * self.d_b[2] * a_aft * a_bft_3 * self.Vft / self.da
        Vtf_b4 = self.ae_b4 * 6 * self.d_b[3] * a_aft * a_bft_4 * self.Vft / self.da

        Vff = self.Vcell - self.Vft

        f[27] = 1 / self.Vft * (self.Qa - Vtf_a)  # tail air mb
        f[28] = 1 / self.Vft * (a_bm_1 * self.Qm - a_bft_1 * self.Qft - Vtf_b1)  # tail mb
        f[29] = 1 / self.Vft * (a_bm_2 * self.Qm - a_bft_2 * self.Qft - Vtf_b2)  # tail mb
        f[30] = 1 / self.Vft * (a_bm_3 * self.Qm - a_bft_3 * self.Qft - Vtf_b3)  # tail mb
        f[31] = 1 / self.Vft * (a_bm_4 * self.Qm - a_bft_4 * self.Qft - Vtf_b4)  # tail mb
        f[32] = 1 / self.Vft * (a_sm_1 * self.Qm - a_sft_1 * self.Qft)  # tail mb
        f[33] = 1 / self.Vft * (a_sm_2 * self.Qm - a_sft_2 * self.Qft)  # tail mb
        f[34] = 1 / self.Vft * (a_sm_3 * self.Qm - a_sft_3 * self.Qft)  # tail mb
        f[35] = 1 / Vff * (Vtf_b1 - a_bff_1 * Qff)  # froth mb
        f[36] = 1 / Vff * (Vtf_b2 - a_bff_2 * Qff)  # froth mb
        f[37] = 1 / Vff * (Vtf_b3 - a_bff_3 * Qff)  # froth mb
        f[38] = 1 / Vff * (Vtf_b4 - a_bff_4 * Qff)  # froth mb
        #################################### UNQUOTE:FT cell MODEL ####################################
        ###############################################################################################

        return f * self.LOWER_LOOP_SAMPLE_TIME
        # return (F * self.LOWER_LOOP_SAMPLE_TIME)



    def reset(self):
        return (np.array(
            self.NormState[:self.Data_Offset.LAST_ACTOR_STATE]))  # (np.array([(self.RR), (0.8)]))# RR_SP = 0.8

    def reset1(self):
        return np.array((self.Reward + 0.5) / (0.5 + 0.5))

    def Get_Norm_States(self):
        return (np.array(self.NormState[:self.Data_Offset.LAST_ACTOR_STATE]))  # States is VF and Bitumen Content

    def Get_States(self):
        return (np.array([(self.RR), (0.8)]))

    def Get_Norm_States_SC(self):
        return np.array((self.Reward + 0.5) / (0.5 + 0.5))


