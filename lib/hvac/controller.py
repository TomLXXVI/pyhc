import numpy as np
import pandas as pd

from nummath import interpolation, graphing


class PIDController:
    def __init__(self, SP, Kr, ti, td, bias, PV_range=(-100.0, 100.0), dt=60.0):
        """
        Set up a PID-controller.
        Params:
            - Kr        proportional gain factor in units of percent span
            - ti        integration time (seconds)
            - td        derivative time (seconds)
            - bias      bias (controller output when deviation is zero) in percent (between 0 and 100 %)
            - SP        set point
            - PV_range  measuring range of the controller
            - dt        sampling time of the controller (seconds)
        """
        self.Kr = Kr
        self.ti = ti
        self.td = td
        self.b = bias

        self.pv = None
        self.PV_min = PV_range[0]
        self.PV_max = PV_range[1]

        self.SP = SP
        # set point in percent of the controller's measuring range
        self.sp = (self.SP - self.PV_min) / (self.PV_max - self.PV_min) * 100.0

        self.dt = dt     # controller's sampling time [s]
        self.e = 0.0     # current error (in percent)
        self.D = 0.0     # value of D control action
        self.reg_D = []  # register for D control action
        self.I = 0.0     # value of I control action
        self.reg_I = []  # register for I control action

    def set_point(self, value):
        """
        Change controller set point.
        """
        self.SP = value
        # set point in percent of the controller's measuring range
        self.sp = (self.SP - self.PV_min) / (self.PV_max - self.PV_min) * 100.0

    def _input(self, t, PV):
        """
        Pass the process value PV at time moment t to the controller.
        """
        # if t is an integer multiple of the controller's sampling time, look for new PV
        if t % self.dt == 0:
            if self.PV_min <= PV <= self.PV_max:
                self.pv = (PV - self.PV_min) / (self.PV_max - self.PV_min) * 100.0
            elif PV < self.PV_min:
                self.pv = 0.0
            else:
                self.pv = 100.0
        # calculate error e(%)
        self.e = self.pv - self.sp

    def _P_action(self):
        """
        Proportional control action.
        """
        return self.Kr * self.e

    def _I_action(self):
        """
        Integrating control action.
        """
        if self.ti != 0.0:
            if len(self.reg_I) < 3:
                # add current deviation e to the I-register
                self.reg_I.append(self.e)
            if len(self.reg_I) == 3:
                # when register contains three e-values, calculate integral with Simpson's rule
                I = (self.reg_I[0] + 4 * self.reg_I[1] + self.reg_I[2]) * (self.dt / 3.0)
                # reset I-register
                self.reg_I = [self.e]
                # accumulated value of I-action
                self.I += (self.Kr / self.ti) * I
            return self.I
        else:
            return 0.0

    def _D_action(self):
        """
        Derivative control action.
        """
        if self.td != 0.0:
            if len(self.reg_D) < 3:
                # add current deviation e to the D-register
                self.reg_D.append(self.e)
            if len(self.reg_D) == 3:
                # when register contains three values, calculate derivative de/dt using
                # backward finite difference approximation of order O(h^2)
                de = (self.reg_D[0] - 4 * self.reg_D[1] + 3 * self.reg_D[2]) / (2 * self.dt)
                # move last two elements to the left in the D-register
                self.reg_D = self.reg_D[-2:]
                # value of D-action
                self.D = self.Kr * self.td * de
            return self.D
        else:
            return 0.0

    def _output(self):
        """
        Return the control output in percent of the control range.
        """
        p = self._P_action()
        i = self._I_action()
        d = self._D_action()
        out = p + i + d + self.b
        if out < 0.0:
            out = 0.0
        elif out > 100.0:
            out = 100.0
        return out

    def __call__(self, t, PV):
        """
        Override () operator.
        Params:
            -   see input method
        Return value:
            -   see output method
        """
        self._input(t, PV)
        return self._output()

    def P_characteristic(self):
        """
        Return the percentual P-control characteristic.
        Abscissa = control output in percent, ordinate = measured process value in percent
        """
        pv = np.linspace(0.0, 100.0, endpoint=True)
        out = self.Kr * (pv - self.sp) + self.b
        return out, pv

    def P_operating_point(self, Kp, Z):
        """
        Calculate the static operating point of the P-controller. This is the intersection of the static control
        characteristic and the static process characteristic for a constant value of the disturbance Z.
        Params:
            - Kp    percentual static gain of the process
            - Z     constant value of disturbance (with the same measuring unit as the process value PV)
        Return value:
            - control output and process value in percent at the intersection
        """
        z = (Z - self.PV_min) / (self.PV_max - self.PV_min) * 100.0
        out = (self.Kr * (z - self.sp) + self.b) / (1 - self.Kr * Kp)
        pv = (z - Kp * self.Kr * self.sp + Kp * self.b) / (1 - self.Kr * Kp)
        return out, pv


class OnOffController:
    def __init__(self, SP, HL_offset, LL_offset, PV_range=(-100.0, 100.0), dt=60.0, ctrl_dir=-1):
        """
        Set up an on-off controller.
        Params:
            - SP            set point
            - HL_offset     high limit offset of dead band with respect to SP
            - LL_offset     low limit offset of dead band with respect to SP
            - PV_range      measuring range of controller
            - dt            sampling time of controller (seconds)
            - ctrl_dir      control direction: -1 = inverse, +1 = direct
        """
        self.SP = SP
        self.HL_offset = HL_offset
        self.LL_offset = LL_offset
        HL = self.SP + HL_offset
        LL = self.SP + LL_offset
        self.PV_range = PV_range
        self.dt = dt
        self.ctrl_dir = ctrl_dir

        self.sp = (SP - PV_range[0]) / (PV_range[1] - PV_range[0]) * 100.0
        hl = (HL - PV_range[0]) / (PV_range[1] - PV_range[0]) * 100.0
        ll = (LL - PV_range[0]) / (PV_range[1] - PV_range[0]) * 100.0
        self.e_ll = ll - self.sp
        self.e_hl = hl - self.sp
        self.pv = None
        self.e = None
        self.out = None

    def set_point(self, value):
        """
        Change controller set point.
        """
        self.SP = value
        HL = self.SP + self.HL_offset
        LL = self.SP + self.LL_offset
        self.sp = (self.SP - self.PV_range[0]) / (self.PV_range[1] - self.PV_range[0]) * 100.0
        hl = (HL - self.PV_range[0]) / (self.PV_range[1] - self.PV_range[0]) * 100.0
        ll = (LL - self.PV_range[0]) / (self.PV_range[1] - self.PV_range[0]) * 100.0
        self.e_ll = ll - self.sp
        self.e_hl = hl - self.sp

    def _input(self, t, PV):
        if t % self.dt == 0:
            self.pv = (PV - self.PV_range[0]) / (self.PV_range[1] - self.PV_range[0]) * 100.0
            self.e = self.pv - self.sp

    def _output(self):
        if self.e <= self.e_ll:
            self.out = 100.0 if self.ctrl_dir == -1 else 0.0
        elif self.e >= self.e_hl:
            self.out = 0.0 if self.ctrl_dir == -1 else 100.0
        # if self.e is between self.e_ll and self.e_hl -> controller output doesn't change
        return self.out

    def __call__(self, t, PV):
        """
        Override () operator.
        Params:
            -   see input method
        Return value:
            -   see output method
        """
        self._input(t, PV)
        return self._output()


class PWMController:
    def __init__(self, SP, Kr, ti, td, bias, n, PV_range=(-100.0, 100.0), dt=60.0):
        """
        Set up a PWM-controller (time proportional controller)
        A PWM-controller is a PID-controller with time proportional on/off-output
        Params:
            - SP            set point
            - Kr            percentual proportional gain of PID controller
            - ti            integration time of PID controller (seconds)
            - td            derivative time of PID controller (seconds)
            - bias          bias of PID controller in percent of control range
            - n             number of sampling time steps that constitutes the PWM cycle period
            - PV_range      measuring range of the PWM controller
            - dt            sampling time of PWM controller (seconds)
        """
        self.T = n * dt         # PWM cycle period is an integer multiple of the controller's sampling time
        self.cycle_output = []  # PWM output cycle
        self.n = n
        self.dt = dt
        self.pid_controller = PIDController(SP, Kr, ti, td, bias, PV_range, dt)

    @property
    def SP(self):
        return self.pid_controller.SP

    @SP.setter
    def SP(self, value):
        """
        Change controller set point.
        """
        self.pid_controller.set_point(value)

    def __call__(self, t, PV):
        # if time t is an integer multiple of the PWM cycle period, a PWM output cycle has been completed
        if t % self.T == 0.0:
            self.cycle_output = [0.0] * self.n          # set up new PWM output cycle
            out = self.pid_controller(t, PV)            # get percentual output from PID controller for current PV
            T_on = (out / 100.0) * self.T               # determine cycle ON time
            n_on = int(T_on // self.dt)                 # determine the number of ON time steps
            self.cycle_output[:n_on] = [100.0] * n_on   # set the next PWM output cycle
            # print(f"time: {t / 3600.0:.3f} - cycle output = {self.cycle_output}")
        # if time t is an integer multiple of the controller's sampling time, return and remove the first element
        # from the PWM output cycle
        if t % self.dt == 0.0:
            return self.cycle_output.pop(0)
        # else only return (but don't remove) the first element
        else:
            return self.cycle_output[0]


class DataLogger:
    def __init__(self, *channel_ids):
        self.header = ['t']
        for channel_id in channel_ids:
            self.header.append(channel_id)
        self.table = []
        self.col_size = len(self.header)
        self.row_size = 0
        self.data = None

    def log(self, time_stamp, **channels):
        row = [None] * self.col_size
        row[0] = time_stamp
        for channel_id, channel_value in channels.items():
            channel_index = self.header.index(channel_id)
            row[channel_index] = channel_value
        self.table.append(row)
        self.row_size += 1

    def get_data(self):
        self.data = pd.DataFrame(data=np.array(self.table), columns=self.header)
        return self.data

    def reset(self):
        self.table = []
        self.row_size = 0
        self.data = None


class OutdoorResetController:
    def __init__(self, c0=None, c1=None, T_min=None, T_max=None):
        """
        Initialize OutdoorResetController with the coefficients of the reset line.
        The reset line is a straight line defining the water entering temperature as function of outdoor temperature:
        T_we = coeffs[0] + coeffs[1] * T_out
        """
        self.T_min = T_min
        self.T_max = T_max
        if None not in [c0, c1]:
            self.c0 = c0
            self.c1 = c1
        else:
            self.c0 = 0.0
            self.c1 = 0.0

    def calc_reset_line(self, Q_load_des, T_out_des, T_we_des, T_in_des, radiator):
        """
        Calculate required reset line from design conditions.
        Params:
        - Q_load_des    design load [W]
        - T_out_des     design outdoor temperature [°C]
        - T_we_des      design water entering temperature [°C]
        - T_in_des      design indoor temperature [°C]
        - radiator      the radiator in the room (Radiator-object)
        """
        K = Q_load_des / (T_in_des - T_out_des)
        V_w_des = radiator.calc_Vw(Q_load_des, T_we_des, T_in_des)
        T_out_array = np.arange(T_out_des, T_in_des + 0.5, 0.5)
        Q_load_array = np.array([K * (T_in_des - T_out_i) for T_out_i in T_out_array])
        T_we_array = np.array([radiator.calc_Twe(Q_load_i, V_w_des, T_in_des) for Q_load_i in Q_load_array])
        coeffs = interpolation.LinReg(T_out_array, T_we_array).solve()
        self.T_min = T_out_des
        self.T_max = T_in_des
        self.c0 = coeffs[0]
        self.c1 = coeffs[1]

    def get_coefficients(self):
        return self.c0, self.c1

    def plot_reset_line(self, fig_size=None, dpi=None):
        graph = graphing.Graph(fig_size=fig_size, dpi=dpi)
        x_data = np.linspace(self.T_min, self.T_max, endpoint=True)
        y_data = self.c0 + self.c1 * x_data
        graph.add_data_set('reset line', x=x_data, y=y_data)
        graph.scale_y_axis(lim_min=int(np.min(y_data)), lim_max=int(np.max(y_data)) + 2, tick_step=2)
        graph.turn_grid_on()
        graph.set_axis_titles(x_title='Tout [°C]', y_title='Twe [°C]')
        graph.draw_graph()
        return graph

    def __call__(self, T_out):
        T_we = self.c0 + self.c1 * T_out
        return T_we
