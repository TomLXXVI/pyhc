import collections

import numpy as np

import nummath.roots


# noinspection PyAttributeOutsideInit
class Room:
    def __init__(self, R_tr=0.0, C_ra=0.0, C_bm=0.0, dt=60.0, radiator=None):
        """
        Create a Room-object.
        """
        # FOR DYNAMIC CALCULATIONS...
        try:
            # thermal characteristics of the room
            self.thermal_characteristics(R_tr, C_ra, C_bm)
            # calculation time step
            self.dt = dt
        except ZeroDivisionError:
            pass

        # INITIAL CONDITIONS
        # list for storing room temperatures at k = -1 and k = -2
        # list for storing building mass temperatures at k = -1 and k = -2
        self.T_r = []
        self.T_bm = []

        # RADIATOR IN THE ROOM
        self.radiator = radiator

        # VALUES AT DESIGN CONDITIONS (USED FOR STATIC ANALYSIS)
        self.Q_e_des = None      # heat load at design conditions
        self.T_out_des = None    # selected outside air temperature at design conditions
        self.T_r_des = None      # desired room air temperature at design conditions
        self.V_w_des = None      # required supply water flow rate at design conditions
        self.T_we_des = None     # selected supply water temperature at design conditions

    # -----------------------------------------------------------------------------------------------------------------
    # METHODS FOR DYNAMIC CALCULATIONS

    def thermal_characteristics(self, R_tr=0.0, C_ra=0.0, C_bm=0.0):
        """
        Set the thermal characteristics of the room (a room is considered as a linear thermal network with two
        temperature nodes: node 'room air' and node 'building mass' (building shell surrounding the room volume).
        Params:
            - R_rm1     convection resistance [K/W] between node 'room air' and node 'building mass'
            - R_rm2     conduction resistance [K/W] between node 'room air' and node 'building mass'
            - R_mo1     conduction resistance [K/W] between node 'building mass' and node 'outside air'
            - R_mo2     convection resistance [K/W] between node 'building mass' and node 'outside air'
            - C_r       thermal capacity [J/K] of room air
            - C_m       effective or diurnal thermal capacity [J/K] of node 'building mass'
        """
        # thermal characteristic values
        self.R_rm = R_tr / 2.0
        self.R_mo = R_tr / 2.0
        self.R = R_tr
        self.C_ra = C_ra
        self.C_bm = C_bm
        self.C = self.C_ra + self.C_bm

        # time constants of the second order process
        x = self.C_ra * (self.R_rm + self.R_mo) + self.C_bm * self.R_mo
        y = np.sqrt(
            1.0
            - (4.0 * self.C_ra * self.R_rm * self.C_bm * self.R_mo)
            / (x ** 2)
        )
        self.tau_1 = 0.5 * x * (1 + y)
        self.tau_2 = 0.5 * x * (1 - y)

    @property
    def dt(self):
        """Return time step used for dynamic calculations."""
        return self._dt

    @dt.setter
    def dt(self, dt):
        """
        Set time step for dynamic calculations.
        (Re)calculate the coefficients of the difference equations.
        """
        self._dt = dt

        # coefficients in difference equations using backward difference approximations of O(h^2)
        # in equation for dTr/dt (change of room air temperature):
        self._a = [
            3.0 + 2.0 * self._dt / (self.R_rm * self.C_ra),  # Tr
            2.0 * self._dt / (self.R_rm * self.C_ra),  # Tm
            2.0 * self._dt / self.C_ra,  # Qe
        ]
        # in equation for dTm/dt (change of building mass temperature):
        self._b = [
            3.0 + 2.0 * self._dt * (self.R_rm + self.R_mo) / (self.R_rm * self.R_mo * self.C_bm),  # Tm
            2.0 * self._dt / (self.R_rm * self.C_bm),  # Tr
            2.0 * self._dt / (self.R_mo * self.C_bm),  # To
        ]

    def initial_values(self, T_r_ini, T_bm_ini):
        """
        Set initial values of Tr and Tm at moment k = -2 and moment k = -1
        Params:
            - Tr_ini    list with the two initial values of Tr at moment k = -2 and moment k = -1
            - Tm_ini    list with the two initial values of Tm at moment k = -2 and moment k = -1
        """
        self.T_r = [T_r_ini[0], T_r_ini[1]]
        self.T_bm = [T_bm_ini[0], T_bm_ini[1]]

    def _calc_Tr(self, T_bm, Q_e):
        # internal method - Calculate and return Tr at current moment k
        return (1.0 / self._a[0]) * (self._a[1] * T_bm + self._a[2] * Q_e - self.T_r[-2] + 4.0 * self.T_r[-1])

    def _calc_Tm(self, T_r, T_out):
        # internal method - Calculate and return Tm at current moment k
        return (1.0 / self._b[0]) * (self._b[1] * T_r + self._b[2] * T_out - self.T_bm[-2] + 4.0 * self.T_bm[-1])

    def _calc_Qe(self, T_r, V_w, T_we):
        # internal method - Calculate and return Qe at current moment k
        return self.radiator.calc_Qe(V_w, T_we, T_r)

    def __call__(self, V_w, T_we, T_out):
        """
        Find room air temperature Tr, building mass temperature Tm and heat output Qe from radiator into room air at
        current time moment k. The calculated results at each time moment are stored in the following lists:
            - 'self.Tr'
            - 'self.Tm'
            - 'self.Qe'
        (Note that at index = 0 and index = 1 these lists contain the initial values set with 'set_initial_values'.)
        Params:
            - Vw    supply water flow rate [m^3/s] through radiator at current time moment k
            - Twe   supply water temperature [°C] at entrance of radiator at current time moment k
            - To    outside air temperature [°C] at current time moment k
        Return value:
            - room air temperature [°C] at current time moment k
            - building mass temperature [°C] at current time moment k
            - heat input from radiator [W] into room air at current time moment k
        """
        # calculation is done by iteration
        tol = 0.1
        T_r = self.T_r[-1]  # initial guess = previous room air temperature
        for i in range(30):
            Q_e = self._calc_Qe(T_r, V_w, T_we)  # calculate heat input from radiator
            T_bm = self._calc_Tm(T_r, T_out)     # calculate building mass temperature
            T_r_old = T_r                        # keep the current guess for comparison with the next estimation
            T_r = self._calc_Tr(T_bm, Q_e)       # calculate new room air temperature
            if abs(T_r - T_r_old) <= tol:
                self.T_r[-2] = self.T_r[-1]
                self.T_r[-1] = T_r
                self.T_bm[-2] = self.T_bm[-1]
                self.T_bm[-1] = T_bm
                return T_r, T_bm, Q_e
        raise OverflowError('too many iterations')

    # -----------------------------------------------------------------------------------------------------------------
    # METHODS FOR STATIC CALCULATIONS

    def design_conditions(self, To_des, Tr_des, Twe_des, Qe_des=0.0, K=0.0):
        """
        Set the design conditions of the room.
        From these design conditions the design water flow rate [m^3/s] through the given radiator is calculated
        (attribute 'self.Vw_des').
        Params:
            - To_des    outside temperature at design conditions [°C]
            - Tr_des    desired room air temperature at design conditions [°C]
            - Twe_des   selected supply water temperature to radiator for design conditions [°C]
            - Qe_des    heat load at design conditions [W]
            - K         global thermal conductance of room [W/K]
        Note:
            Either 'Qe_des' or 'K' must be set. It is not necessary to include both.
        """
        self.T_out_des = To_des
        self.T_r_des = Tr_des
        self.T_we_des = Twe_des
        if not Qe_des:
            self.Q_e_des = (self.T_r_des - self.T_out_des) / K
            self.R = 1.0 / K
        else:
            self.Q_e_des = Qe_des
            self.R = (self.T_r_des - self.T_out_des) / self.Q_e_des
        self.V_w_des = self.radiator.calc_Vw(self.Q_e_des, self.T_we_des, self.T_r_des)

    def calc_static_Tr(self, V_w, T_we, T_out):
        """
        Find room air temperature 'Tr' in stationary regime for a given:
            - flow rate through the radiator,
            - supply water temperature,
            - and a given outside temperature.
        Params:
            - Vw    constant supply water flow rate [m^3/s] through radiator
            - Twe   constant supply water temperature [°C] at entrance of radiator
            - To    constant outside air temperature [°C]
        Return value:
            - stationary room air temperature [°C]
        Note:
            Before calling this method, two parameters must be known:
            - the design outside temperature must have been set using method 'set_design_conditions' or '.To_des = ...'
            - the global thermal resistance of the room must have been calculated using method 'set_design_conditions'
              or must have been set using '.R = ...'
        """
        # Find Tr for which the expression:
        #   To + (k/K) * [ (Twe - Tr)^2 - (K / (rho * c * Vw)) * (Tr - To) * (Twe - Tr) ]^(n/2) - Tr equals zero
        def f(T_r):
            k = self.radiator.k
            K = 1.0 / self.R
            rho = self.radiator.water['rho']
            c = self.radiator.water['c']
            n = self.radiator.n
            t = (T_we - T_r) ** 2 - (K / (rho * c * V_w)) * (T_r - T_out) * (T_we - T_r)
            return T_out + (k / K) * t ** (n / 2.0) - T_r if t > 0.0 else -1.0

        # search for a root in the interval Tr = [To_des, 50.0] °C using a initial search step of 5 °C
        zeros = nummath.roots.FunctionRootSolver(f, [self.T_out_des, 50.0], 5).solve()
        return zeros[0]  # return the first root (if multiple roots should be found)

    def static_characteristic(self, T_out, T_we=None):
        """
        Calculate the static room characteristic 'Tr = f(Vw(%))' or 'Tr = f(Qe(%))' with Vw and Qe in percent of
        maximum values (i.e. when the radiator valve is fully open).
        Params:
            -   the constant outside air temperature [°C]
            -   the constant supply water temperature to the radiator [°C] (default value if 'Twe' is left None,
                is the design value set via method 'set_design_conditions')
        Return values:
            -   array with the percentual flow rates from 0 to 100 % of the design flow rate 'self.Vw_des'
                (calculated from the design conditions when the method 'set_design_conditions' is called).
            -   array with the resulting percentual heat outputs of the radiator from 0 to 100 %
            -   array with the resulting room temperatures [°C]
            -   the maximum heat output of the radiator [W] at 100 % flow rate (i.e. design flow rate)
        Note:
            Before calling this method, two parameters must be known:
            - the design flow rate through the radiator, using the method 'set_design_conditions' or by setting the
              attribute '.Vw_des'
            - the supply water temperature if param 'Twe' is left None, using the method 'set_design_conditions' or
              by setting the attribute '.Twe_des'
        """
        # set up a range of flow rates from 0 % to 100 % of design flow rate
        V_w_percent = np.linspace(1.0e-12, 100.0, endpoint=True)
        V_w = 0.01 * V_w_percent * self.V_w_des
        # calculate the stationary room temperature for each flow rate
        if T_we is None: T_we = self.T_we_des
        T_r = np.array([self.calc_static_Tr(Vw_i, T_we, T_out) for Vw_i in V_w])
        # calculate the stationary heat input = stationary heat loss at each flow rate
        Q_e = np.array([self.radiator.calc_Qe(Vw_i, T_we, Tr_i) for Vw_i, Tr_i in zip(V_w, T_r)])
        Q_e_percent = Q_e / Q_e[-1] * 100.0
        return V_w_percent, Q_e_percent, T_r, Q_e[-1]


class DeadTime:
    """
    Class for adding dead time to a control loop.
    """
    def __init__(self, n, init_value):
        """
        Set up a dead time using a FIFO-buffer .
        Params:
            - n             the number of time steps dt a value is kept in the buffer
            - init_value    initial values in the buffer at the start of the control loop
        """
        self._buffer = collections.deque([init_value] * n)

    def input(self, value):
        """
        Add new value at the head of the buffer.
        """
        self._buffer.appendleft(value)

    def output(self):
        """
        Get and remove buffered value from the tail of the buffer
        """
        return self._buffer.pop()

    def __call__(self, value):
        """
        Add new value at the head of the buffer and get and remove the value at the tail of the buffer.
        """
        self.input(value)
        return self.output()


class OutsideTemperature:
    def __init__(self, profile='sine', **kwargs):
        if profile == 'sine':
            self.avg = kwargs['avg']
            self.ampl = kwargs['ampl']
            self.period = kwargs['period']
            self._profile = self._sine
            self.w = 2.0 * np.pi / self.period
        if profile == 'trapezium':
            self.min = kwargs['min']
            self.max = kwargs['max']
            self.dt_rise = kwargs['dt_rise']
            self.dt_fall = kwargs['dt_fall']
            self.period = kwargs['period']
            self._profile = self._trapezium
            self.k_up = (self.max - self.min) / self.dt_rise
            self.k_down = (self.min - self.max) / self.dt_fall

    def __call__(self, t):
        return self._profile(t)

    def _trapezium(self, t):
        n = t // self.period
        t = t - n * self.period
        if t == 0:
            return self.min
        elif 0 < t <= self.dt_rise:
            return self.min + self.k_up * t
        elif self.dt_rise < t <= (self.period - self.dt_fall):
            return self.max
        elif (self.period - self.dt_fall) < t <= self.period:
            return self.max + self.k_down * (t - (self.period - self.dt_fall))
        else:
            return self.min

    def _sine(self, t):
        return self.avg + self.ampl * np.sin(self.w * t)
