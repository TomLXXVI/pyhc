import numpy as np

import nummath.roots as roots


class Radiator:
    def __init__(self, Qe_nom, Twe_nom, Twl_nom, Tr_nom, n, rho_w=1000.0, c_w=4186.0):
        """
        Configure radiator from catalog data.
        Params:
            - Qe_nom    heat output at nominal conditions [W]
            - Twe_nom   nominal water temperature at entrance [°C]
            - Twl_nom   nominal water temperature at exit [°C]
            - Tr_nom    nominal room temperature [°C]
            - n         radiator exponent [-]
            - rho_w     mass density of water [kg/m^3]
            - c_w       specific heat of water [J/(kg.K)]
        """
        self._Qe_nom = Qe_nom
        self._Twe_nom = Twe_nom
        self._Twl_nom = Twl_nom
        self._Tr_nom = Tr_nom
        self._n = n
        self._rho_w = rho_w
        self._c_w = c_w
        # max. possible water temperature difference between entrance and exit
        delta_Tw_max = self._Twe_nom - self._Tr_nom
        # nom. water temperature difference between entrance and exit
        delta_Tw_nom = self._Twe_nom - self._Twl_nom
        # radiator overall heat transfer coefficient [W/K]
        self._k = self._Qe_nom / (delta_Tw_max ** 2 - delta_Tw_max * delta_Tw_nom) ** (0.5 * self._n)
        # flow rate at nominal conditions [m^3/s]
        self._Vw_nom = self._Qe_nom / (self._rho_w * self._c_w * delta_Tw_nom)
        # actual flow rate [m^3/s], entering water temperature [°C], leaving water temperature [°C]
        # and room temperature [°C]
        self._Vw = None
        self._Twe = None
        self._Twl = None
        self._Tr = None

    @property
    def nominal_specs(self):
        """
        Return nominal parameters of radiator as dict.
        Keywords:
            - 'Qe'      nominal heat output
            - 'Twe'     nominal entering water temperature
            - 'Twl'     nominal leaving water temperature
            - 'Tr'      nominal room temperature
            - 'Vw'      nominal flow rate
        """
        return {
            'Qe':   self._Qe_nom,
            'Twe':  self._Twe_nom,
            'Twl':  self._Twl_nom,
            'Tr':   self._Tr_nom,
            'Vw':   self._Vw_nom
        }

    @property
    def Vw(self):
        """Return actual flow rate [m^3/s]."""
        return self._Vw

    @Vw.setter
    def Vw(self, Vw):
        """Set actual flow rate [m^3/s]."""
        self._Vw = Vw

    @property
    def Twe(self):
        """Return actual entering water temperature [°C]."""
        return self._Twe

    @Twe.setter
    def Twe(self, Twe):
        """Set actual water entering temperature [°C]."""
        self._Twe = Twe

    @property
    def Tr(self):
        """Return actual room temperature [°C]."""
        return self._Tr

    @Tr.setter
    def Tr(self, Tr):
        """Set actual room temperature [°C]."""
        self._Tr = Tr

    @property
    def Twl(self):
        """Return actual leaving water temperature [°C]."""
        return self._Twl

    @property
    def n(self):
        """Return radiator exponent."""
        return self._n

    @property
    def k(self):
        """Return radiator overall heat transfer coefficient [W/K]."""
        return self._k

    @property
    def water(self):
        """
        Return radiator water properties as a dict
        Keywords:
        - 'rho'     mass density of water [kg/m^3]
        - 'c'       specific heat of water [J/(kg.K)]
        """
        return {
            'rho': self._rho_w,
            'c': self._c_w
        }

    def calc_Qe(self, Vw, Twe, Tr):
        """
        Calculate emitted heat from radiator for given water flow rate, entering water temperature and room temperature.
        Params:
            - Vw       flow rate of water through radiator [m^3/s]
            - Twe      entering water temperature [°C]
            - Tr       room temperature [°C]
        Return value:
            - Heat output from radiator [W]
        """
        if Vw != 0.0:
            e = 1 / (0.5 * self._n)
            delta_Tw_max = Twe - Tr

            def f(x):
                return (self._k ** e * delta_Tw_max ** 2 -
                        self._k ** e * delta_Tw_max * (x / (self._rho_w * self._c_w * Vw)) - x ** e)

            # theoretical maximum heat output (delta_Tw = 0)
            Qe_max = self._k * delta_Tw_max**self._n
            zeros = roots.FunctionRootSolver(f, [0, Qe_max], 100.0).solve()
            Qe = zeros[0]
            # calc. leaving water temperature
            self._Twl = Twe - Qe / (self._rho_w * self._c_w * Vw)
            return Qe
        else:
            return 0.0

    def calc_Vw(self, Qe, Twe, Tr):
        """
        Calculate flow rate needed to establish the required heat output from the radiator in order to maintain
        the given room temperature when entering water temperature is given.
        Params:
            - Qe    the demanded heat output [W]
            - Twe   the entering water temperature [°C]
            - Tr    the room temperature [°C]
        Return value:
            - Flow rate through radiator [m^3/s]
        """
        if Qe != 0.0:
            delta_Tw_max = Twe - Tr
            delta_Tw = delta_Tw_max - (Qe / self._k)**(1/(0.5 * self._n)) / delta_Tw_max
            if delta_Tw <= 0.0:
                raise ValueError(f"entering water temperature of {Twe} °C is too low.")
            Vw = Qe / (self._rho_w * self._c_w * delta_Tw)
            # calc. leaving water temperature
            self._Twl = Twe - Qe / (self._rho_w * self._c_w * Vw)
            return Vw
        else:
            return 0.0

    def calc_Twe(self, Qe, Vw, Tr):
        """
        Calculate required entering water temperature to establish the given heat output from the radiator in order to
        maintain the given room temperature when water flow rate is given.
        Params:
            - Qe    the demanded heat output [W]
            - Vw    the water flow rate through the radiator [m^3/s]
            - Tr    the room temperature [°C]
        Return value:
            - Entering water temperature [°C]
        """
        delta_Tw = Qe / (self._rho_w * self._c_w * Vw)
        a = 1
        b = -delta_Tw
        c = -(Qe / self._k) ** (1/(0.5 * self._n))
        D = b ** 2 - 4 * a * c
        delta_Tw_max = max((-b + np.sqrt(D)) / (2 * a), (-b - np.sqrt(D)) / (2 * a))
        return Tr + delta_Tw_max

    def calc_Vw_primary(self, Qe, Vw, Tr, Twe_d):
        """
        Calculate in case of mixing the primary water flow rate to the mixing valve that is needed to establish the
        given heat output in order to maintain the given room temperature when constant water flow rate through the
        radiator is given.
        Params:
            - Qe        demanded heat output [W]
            - Vw        water flow rate through the radiator [m^3/s]
            - Tr        room temperature [°C]
            - Twe_d     required entering water temperature at design conditions
        Return value:
            - Primary flow rate towards mixing point [m^3/s]
        """
        Twe = self.calc_Twe(Qe, Vw, Tr)
        delta_Tw = Qe / (self._rho_w * self._c_w * Vw)
        Twl = Twe - delta_Tw
        return (delta_Tw / (Twe_d - Twl)) * Vw

    def characteristic(self, Vw_per=np.linspace(0.0, 100.0, endpoint=True)):
        """
        Calculate radiator characteristic Qe(%) = f(Vw(%)).
        For the method to work the actual supply water temperature, the actual room temperature and
        maximum water flow rate must be specified first through the appropriate properties: see 'self.Twe', 'self.Tr',
        and 'self.Vw' (with Vw in this case the maximum flow rate, ie. with the radiator valve fully open).
        Params:
            - Vw_per    array with percentage flow rates
        Return values:
            - array with corresponding percentage heat outputs
            - the heat output at maximum flow rate 'Vw_max' [W]
        """
        Vw_max = self._Vw
        Vw = (Vw_per / 100.0) * Vw_max
        Qe = np.array([self.calc_Qe(Vw_i, self._Twe, self._Tr) for Vw_i in Vw])
        Qe_per = Qe / Qe[-1] * 100.0  # Qe[-1] = max. heat output at 100 % flow rate for given 'Tr' and 'Twe'
        return Qe_per, Qe[-1]


class ControlValve:
    def __init__(self, a, R, V_max, type_='equal percentage'):
        """
        Setup a control valve.
        Params:
            - a      valve authority
            - R      inherent rangeability (ratio of maximum to minimum controllable flow rate)
            - V_max  flow rate through the fully open control valve [m^3/s]
            - type_  type of control valve; possible values: 'linear' and 'equal percentage' (default)
        """
        self._a = a
        self._R = R
        self._type = type_
        self._h = 0.0
        self._V_max = V_max

    def input(self, h_per):
        """
        Set valve position in percent of full stem travel (0...100 %).
        """
        self._h = h_per / 100.0

    def output(self):
        """
        Return flow rate through control valve [m^3/s].
        """
        if self._type == 'linear':
            A_rel = self._h
        else:
            A_rel = (1 / self._R) * np.exp(np.log(self._R) * self._h)
        n = (1 / self._a) * A_rel ** 2
        d = 1 + ((1 - self._a) / self._a) * A_rel ** 2
        V_rel = np.sqrt(n / d)
        return V_rel * self._V_max

    def __call__(self, h_per):
        self.input(h_per)
        return self.output()

    def characteristic(self, h_per=np.linspace(0.0, 100.0, endpoint=True)):
        """
        Get installed control valve characteristic.
        Params:
            - h_per     array with valve plug positions in percent
        Return value:
            - array with flow rates in percent of maximum flow rate
        """
        Vw = np.array([self(h_i) for h_i in h_per])
        return (Vw / self._V_max) * 100


# noinspection PyAttributeOutsideInit
class ControlValveMotor:
    def __init__(self, travel_speed, dt=60.0):
        """
        Initialize ControlValveMotor-object.
        Params:
        - travel_speed  valve stem travel speed [% / s]
        - dt            calculation time step (time duration between 2 control loop evaluations) [s]
        """
        self.travel_speed = travel_speed
        self.dt = dt
        # actual valve stem position in % of total travel
        self.h = 0.0

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        # max. valve stem travel during 1 time step 'dt'
        self.travel_limit = self.travel_speed * dt

    def __call__(self, out):
        # required valve stem travel to reach the commanded valve stem position 'out'
        req_travel = out - self.h
        # if the required valve stem travel is smaller than or equal to the travel limit:
        #   the commanded valve position 'out' will be reached within one calculation time step 'dt',
        # else:
        #   the new valve position reached is equal to the actual position + the valve stem travel limit.
        if abs(req_travel) <= self.travel_limit:
            self.h = out
        else:
            self.h = self.h + self.travel_limit if out > self.h else self.h - self.travel_limit
        return self.h
