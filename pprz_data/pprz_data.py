from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, interp1d
from pprz_data import pprz_message_definitions as msg

import pdb


class DATA:
    """
    Data class from Paparazzi System.
    """

    def __init__(self, filename=None, ac_id=None, data_type=None, pad=10, sample_period=0.01):
        self.df_list = []
        self.filename = filename
        self.ac_id = ac_id
        self.df = None
        self.data_values = 0.
        self.data_type = data_type
        self.pad = pad
        self.sample_period = sample_period
        if self.data_type == 'fault':
            self.read_msg1_bundle()
        elif self.data_type == 'flight':
            self.read_msg1_bundle()
            self.read_msg2_bundle()
            self.read_msg3_bundle()
        elif self.data_type == 'robust':
            self.read_msg1_bundle()
            self.read_msg2_bundle()
            self.read_msg3_bundle()
            self.read_msg4_bundle()
        elif self.data_type == 'replay':
            self.read_replay_msg_bundle()

        self.find_min_max()
        self.df_All = self.combine_dataframes()

    def read_msg1_bundle(self):
        try:
            msg_name = 'attitude';
            columns = ['time', 'phi', 'psi', 'theta'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Attitude msg doesnt exist ')
        try:
            msg_name = 'mode';
            columns = ['time', 'mode', '1', '2', '3', '4', '5'];
            drop_columns = ['time', '1', '2', '3', '4', '5']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print('Paparazzi Mode msg doesnt exist ')
        try:
            msg_name = 'imuaccel';
            columns = ['time', 'Ax', 'Ay', 'Az'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' IMU Acceleration msg doesnt exist ')
        try:
            msg_name = 'imuaccel_scaled';
            columns = ['time', 'Ax_sca', 'Ay_sca', 'Az_sca'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' IMU Scaled Acceleration msg doesnt exist ')
        try:
            msg_name = 'imuaccel_raw';
            columns = ['time', 'Ax_raw', 'Ay_raw', 'Az_raw'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' IMU Raw Acceleration msg doesnt exist ')
        try:
            msg_name = 'gps';
            columns = ['time', '1', 'east', 'north', 'course', 'alt', 'vel', 'climb', '8', '9', '10', '11'];
            drop_columns = ['time', '1', '8', '9', '10', '11']
            df = self.extract_message(msg_name, columns, drop_columns)
            df.alt = df.alt / 1000.
            df.vel = df.vel / 100.  # convert to m/s
            df.climb = df.climb / 100.  # convert to m/s
            print(' Generating 3D velocity...')
            df['vel_3d'] = df.climb.apply(lambda x: x ** 2)
            df.vel_3d = df.vel_3d + df.vel.apply(lambda x: x ** 2)
            df.vel_3d = df.vel_3d.apply(lambda x: np.sqrt(x))
            #             if 1:
            #                 # Calculate 3D speed (including the vertical component to the horizontal speed on ground.)
            #                 print(' Calculating the 3D speed norm !')
            #                 df['vel_3d1'] = df.climb.apply(lambda x: x**2)
            #                 print(df.vel_3d1.any())
            self.df_list.append(df)
        except:
            print(' GPS msg doesnt exist ')
        try:
            msg_name = 'imugyro';
            columns = ['time', 'Gx', 'Gy', 'Gz'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' IMU Gyro msg doesnt exist ')
        try:
            msg_name = 'imugyro_scaled';
            columns = ['time', 'Gx_sca', 'Gy_sca', 'Gz_sca'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' IMU Scaled Gyro msg doesnt exist ')
        try:
            msg_name = 'imugyro_raw';
            columns = ['time', 'Gx_raw', 'Gy_raw', 'Gz_raw'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' IMU Raw Gyro msg doesnt exist ')
        try:
            msg_name = 'fault_telemetry';
            columns = ['time', 'Fault_Telemetry'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Fault Telemetry msg doesnt exist ')

    def read_msg2_bundle(self):
        try:
            msg_name = 'actuators';
            columns = ['time', 'S0', 'S1', 'S2'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Actuators msg doesnt exist ')
        try:
            msg_name = 'commands';
            columns = ['time', 'C0', 'C1', 'C2'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Commands msg doesnt exist ')
        try:
            msg_name = 'energy_new';
            columns = ['time', 'Throttle', 'Volt', 'Amp', 'Watt', 'mAh', 'Wh'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Energy_new msg doesnt exist ')
        try:
            msg_name = 'air_data';
            columns = ['time', 'Ps', 'Pdyn_AD', 'temp', 'qnh', 'amsl_baro', 'airspeed', 'TAS'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Air Data msg doesnt exist ')
        try:
            msg_name = 'desired';
            columns = ['time', 'D_roll', 'D_pitch', 'D_course', 'D_x', 'D_y', 'D_altitude', 'D_climb', 'D_airspeed'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Desired msg doesnt exist ')
        try:
            msg_name = 'actuators_4';
            columns = ['time', 'M1_pprz', 'M2_pprz', 'M3_pprz', 'M4_pprz'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' 4-valued Actuators msg doesnt exist ')

    def read_msg3_bundle(self):
        try:
            msg_name = 'gust';
            columns = ['time', 'wx', 'wz', 'Va_gust', 'gamma_gust', ' AoA_gust', 'theta_com_gust'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Gust msg does not exist ')
        # <message name="SOARING_TELEMETRY" id="212">
        # <field name="velocity"     type="float"  unit="m/s">veocity</field>
        # <field name="a_attack"     type="float"  unit="rad">angle of attack</field>
        # <field name="a_sideslip"   type="float"  unit="rad">sideslip angle</field>
        # <field name="dynamic_p"    type="float"  unit="Pa"/>
        # <field name="static_p"     type="float"  unit="Pa"/>
        # <field name="wind_x"       type="float"  unit="m/s"/>
        # <field name="wind_z"       type="float"  unit="m/s"/>
        # <field name="wind_x_dot"   type="float"  unit="m/s2"/>
        # <field name="wind_z_dot"   type="float"  unit="m/s2"/>
        # <field name="wind_power"   type="float"  unit="W"/>
        try:
            msg_name = 'soaring_telemetry';
            columns = ['time', 'sp_Va', 'sp_aoa', 'sp_beta', 'sp_dyn_p', 'sp_sta_p', 'sp_wx', 'sp_wz', 'sp_d_wx',
                       'sp_d_wz', 'sp_w_power'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Soaring Telemetry msg does not exist ')
        # <message name="ROTORCRAFT_FP" id="147">
        #   <field name="east"     type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="north"    type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="up"       type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="veast"    type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
        #   <field name="vnorth"   type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
        #   <field name="vup"      type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
        #   <field name="phi"      type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
        #   <field name="theta"    type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
        #   <field name="psi"      type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
        #   <field name="carrot_east"   type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="carrot_north"  type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="carrot_up"     type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="carrot_psi"    type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
        #   <field name="thrust"        type="int32"/>
        #   <field name="flight_time"   type="uint16" unit="s"/>
        # </message>
        try:
            msg_name = 'rotorcraft_fp';
            columns = ['time', 'east', 'north', 'up', 'veast', 'vnorth', 'vup', 'phi', 'theta', 'psi', 'carrot_east',
                       'carrot_north', 'carrot_up', 'carrot_psi', 'thrust', 'flight_time'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Rotorcraft_fp msg does not exist ')

    def read_msg4_bundle(self):
        try:
            '''This one is a bit hardcoded !!! Sorry ! '''
            motor_df_list = msg.read_log_dshot_telemetry(self.ac_id, self.filename)
            for df in motor_df_list:
                self.df_list.append(df)
        except:
            print(' DSHOT TELEMETRY msg does not exist ')
        try:
            msg_name = 'payload6';
            columns = ['time', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' PAYLOAD6 msg does not exist ')
        try:
            msg_name = 'actuators_4';
            columns = ['time', 'M1_pprz', 'M2_pprz', 'M3_pprz', 'M4_pprz'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' 4-valued Actuators msg doesnt exist ')
        try:
            msg_name = 'actuators_6';
            columns = ['time', 'M1_pprz', 'M2_pprz', 'M3_pprz', 'M4_pprz', 'M5_pprz', 'M6_pprz'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' 6-valued Actuators msg doesnt exist ')
        try:
            msg_name = 'actuators_8';
            columns = ['time', 'S1', 'S2', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' 8-valued Actuators msg doesnt exist ')
        try:
            msg_name = 'rotorcraft_fault';
            columns = ['time', 'M1F', 'M2F', 'M3F', 'M4F', 'M5F', 'M6F'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' ROTORCRAFT FAULT msg does not exist (hexa-version)')
        try:
            msg_name = 'adc_consumptions';
            columns = ['time', 'Pow1', 'Pow2'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' ADC_CONSUMPTIONS msg does not exist ')
        try:
            msg_name = 'robust_morph_angle';
            columns = ['time', 'Morph1', 'Morph2'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' MORPH_ANGLE msg does not exist (This is for RoBust-Morphing-Hexa)')

    def read_replay_msg_bundle(self):
        try:
            msg_name = 'rotorcraft_fp';
            columns = ['time', 'east', 'north', 'up', 'veast', 'vnorth', 'vup', 'phi', 'theta', 'psi', 'carrot_east',
                       'carrot_north', 'carrot_up', 'carrot_psi', 'thrust', 'flight_time'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            print(' Rotorcraft_fp msg does not exist ')
        try:
            msg_name = 'robust_morph_angle';
            columns = ['time', 'Morph1', 'Morph2'];
            drop_columns = ['time']
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))
        except:
            pr

    def get_settings(self):
        ''' Special Message used for the fault injection settings
        2 multiplicative, and 2 additive, and only appears when we change them
        so the time between has to be filled in...'''
        msg_name = 'settings';
        columns = ['time', 'm1', 'm2', 'add1', 'add2'];
        drop_columns = ['time']
        df = self.extract_message(msg_name, columns, drop_columns)
        df.add1 = df.add1 / 9600.;
        df.add2 = df.add2 / 9600.
        return df

    def extract_message(self, msg_name, columns, drop_columns):
        ''' Given msg names such as attitute, we will call msg.read_log_attitute'''
        exec('self.data_values = msg.read_log_{}(self.ac_id, self.filename)'.format(msg_name))
        df = pd.DataFrame(self.data_values, columns=columns)
        df.index = df.time
        df.drop(drop_columns, axis=1, inplace=True)
        return df

    def find_min_max(self):
        self.min_t = 1000.
        self.max_t = -1.
        for df in self.df_list:
            self.min_t = min(self.min_t, min(df.index))
            self.max_t = max(self.max_t, max(df.index))
        print('Min time :', self.min_t, 'Maximum time :',
              self.max_t)  # Minimum time can be deceiving... we may need to find a better way.

    def linearize_time(self, df, min_t=None, max_t=None):
        if (min_t or max_t) == None:
            min_t = min(df.index)
            max_t = max(df.index)
        time = np.arange(int(min_t) + self.pad, int(max_t) - self.pad, self.sample_period)
        out = pd.DataFrame()
        out['time'] = time
        for col in df.columns:
            func = interp1d(df.index, df[col],
                            fill_value='extrapolate')  # FIXME : If we want to use a different method other than linear interpolation.
            out[col] = func(time)
        out.index = out.time
        out.drop(['time'], axis=1, inplace=True)
        return out

    def combine_dataframes(self):
        frames = [self.linearize_time(df, self.min_t, self.max_t) for df in self.df_list]
        return pd.concat(frames, axis=1, ignore_index=False, sort=False)

    def combine_settings_dataframe(self):
        df_settings = self.get_settings()
        df = self.df_All.copy()
        df["m1"] = np.nan;
        df["m2"] = np.nan;
        df["add1"] = np.nan;
        df["add2"] = np.nan
        # Starting time
        i_st = df.index[0]
        for i in range(len(df_settings.index)):
            row_idx = int(round(df_settings.index[i] - i_st, 1) / self.sample_period)
            df.loc[df.index[row_idx], 'm1'] = df_settings.m1.iloc[i]
            df.loc[df.index[row_idx], 'm2'] = df_settings.m2.iloc[i]
            df.loc[df.index[row_idx], 'add1'] = df_settings.add1.iloc[i]
            df.loc[df.index[row_idx], 'add2'] = df_settings.add2.iloc[i]
        return df

    def get_labelled_data(self):
        df = self.combine_settings_dataframe()

        first_idx = df.index[0]
        df.loc[first_idx, 'm1'] = 1.0
        df.loc[first_idx, 'm2'] = 1.0
        df.loc[first_idx, 'add1'] = 0.0
        df.loc[first_idx, 'add2'] = 0.0

        return df.ffill()

    # def combine_settings_dataframe(self):
    #     df_settings = self.get_settings() #FIXME : we may check if this has been already done before or not...
    #     return pd.concat(([self.df_All, df_settings]), axis=1, ignore_index=False, sort=False)

    # def get_labelled_data(self):
    #     df = self.combine_settings_dataframe()
    #     df.m1.iloc[0] = 1.0
    #     df.m2.iloc[0] = 1.0
    #     df.add1.iloc[0] = 0.0
    #     df.add2.iloc[0] = 0.0
    #     return df.ffill()