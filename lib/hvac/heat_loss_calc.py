"""Calculation of heat loss by transmission"""

import cmath
import numpy as np
from nummath import linearsystem


class BuildingElement:
    def __init__(self, **kwargs):
        """
        Configuration of building element.
        Possible values for 'kwargs':
        - A         area of building element [m^2]
        - t         thickness of building element [m]
        - k         coefficient of conduction of building material [W/(m.K)]
        - rho       mass density of building material [kg/m^3]
        - cm        specific heat capacity of building material [J/(kg.K)]
        - r         specific thermal resistance of building element [(m^2.K)/W]
        - ca        specific thermal capacity [J/(m^2.K)]
        - corr_r    correction term to be added to 'r' [(m^2.K)/W]
        """
        self._A = kwargs['A'] if 'A' in kwargs.keys() else 1.0
        self._t = kwargs['t'] if 't' in kwargs.keys() else 0.0
        self._k = kwargs['k'] if 'k' in kwargs.keys() else 0.0
        self._rho = kwargs['rho'] if 'rho' in kwargs.keys() else 0.0
        self._cm = kwargs['cm'] if 'cm' in kwargs.keys() else 0.0
        self._corr_r = kwargs['corr_r'] if 'corr_r' in kwargs.keys() else 0.0
        self._r = kwargs['r'] if 'r' in kwargs.keys() else 0.0
        self._ca = kwargs['ca'] if 'ca' in kwargs.keys() else 0.0

        # used for calculation of effective thermal capacity of BuildingPart: the number of layers the building element
        # will be divided into
        self.num_of_layers = 1.0

    @property
    def r(self):
        """Specific thermal resistance of building element [(m^2.K)/W]"""
        if not self._r:
            self._r = self._t / self._k + self._corr_r
        return self._r

    @property
    def u(self):
        """Specific thermal conductance of building element [W/(m^2.K)]"""
        return 1.0 / self.r

    @property
    def U(self):
        """Thermal conductance of building element [W/K]"""
        return self.u * self._A

    @property
    def ca(self):
        """Specific thermal capacity of building element [J/(m^2.K)]"""
        if not self._ca:
            self._ca = self._rho * self._cm * self._t
        return self._ca

    @property
    def C(self):
        """Thermal capacity of building element [J/K]"""
        return self.ca * self._A

    @property
    def A(self):
        """Area of building element [m^2]"""
        return self._A

    @property
    def t(self):
        """Thickness of building element [m]"""
        return self._t


class BuildingCompositeElement:
    """Multiple building elements that are parallel connected (in the same plane)"""
    def __init__(self, *building_elements):
        self.building_elements = building_elements
        self._r = 0.0
        self._ca = 0.0

        # used for calculation of effective thermal capacity of BuildingPart: the number of layers the building element
        # will be divided into
        self.num_of_layers = 1.0

    @property
    def r(self):
        """Specific thermal resistance of composite building element [(m^2.K)/W]"""
        A = 0.0
        U = 0.0
        for elem in self.building_elements:
            A += elem.A
            U += elem.U
        self._r = A / U
        return self._r

    @property
    def ca(self):
        """Specific thermal capacity of composite building element [J/(m^2.K)]"""
        A = 0.0
        C = 0.0
        for elem in self.building_elements:
            A += elem.A
            C += elem.C
        self._ca = C / A
        return self._ca


class BuildingPart:
    """Multiple building elements in series."""
    def __init__(self, *building_elements, corr_u=0.0, r_conv_in=0.0, r_conv_out=0.0):
        self.building_elements = building_elements
        self._r_conv_in = r_conv_in     # spec. convection resistance at inside surface of building part [(m^2.K)/W]
        self._r_conv_out = r_conv_out   # spec. convection resistance at outside surface of building part [(m^2.K)/W]
        self._r = 0.0                   # spec. thermal resistance of building part [(m^2.K)/W]
        self._ca = 0.0                  # spec. thermal capacity per unit area of building part [J/(m^2.K)]
        self._ca_eff = 0.0              # spec. thermal effective capacity per unit area of building part [J/(m^2.K)]
        self._t = 0.0                   # total thickness of building part [m]
        self._corr_u = corr_u           # correction term for u-value
        self.T_in = 0.0                 # temperature at inside of building part [°C]
        self.T_out = 0.0                # temperature at outside of building part [°C]
        self.A_in = 0.0                 # inside surface of building part [m^2]
        self.A_out = 0.0                # outside surface of building part [m^2]

    def set_temperatures(self, T_inside, T_outside):
        """Set inside and outside temperature at both sides of building part [°C]"""
        self.T_in = T_inside
        self.T_out = T_outside

    def set_areas(self, A_inside, A_outside):
        """Set inside and outside area of building part [m^2]"""
        self.A_in = A_inside
        self.A_out = A_outside

    @property
    def r(self):
        """Specific thermal resistance of building part [(m^2.K)/W]"""
        self._r = self._r_conv_in
        for elem in self.building_elements:
            self._r += elem.r
        self._r = 1.0 / (1.0 / self._r + self._corr_u)
        self._r += self._r_conv_out
        return self._r

    @property
    def u(self):
        """Specific thermal conductance of building part [W/(m^2.K)]"""
        return 1.0 / self.r

    @property
    def ca(self):
        """Specific thermal capacity of building part [J/(m^2.K)]"""
        self._ca = 0.0
        for elem in self.building_elements:
            self._ca += elem.ca
        return self._ca

    def calculate_effective_capacity(self, T_out_ampl=5.0, T_out_period=24.0):
        """Calculate effective thermal capacity of building part"""
        layers = []
        for be in self.building_elements:
            r_layer = be.r / be.num_of_layers
            ca_layer = be.ca / be.num_of_layers
            layers.append(BuildingElement(r=r_layer, ca=ca_layer))

        r = [self._r_conv_out + 0.5 * layers[0].r]
        for i in range(len(layers)):
            r.append(0.5 * (layers[i - 1].r + layers[i].r))
        r.append(0.5 * layers[-1].r + self._r_conv_in)

        w = 2.0 * cmath.pi / (T_out_period * 3600.0)
        xc = [1.0 / (w * layers[i].ca * 1.0j) for i in range(len(layers))]

        To = cmath.rect(T_out_ampl, -cmath.pi / 2.0)

        n = len(layers)
        A = np.zeros((2 * n + 1, 2 * n + 1), dtype=complex)
        for i, j in zip(range(0, n + 1), range(0, 2 * n + 1, 2)):
            A[i, j] = -r[i]
            if j < 2 * n:
                A[i, j + 1] = -xc[i]
            if i > 0 and j > 0:
                A[i, j - 1] = xc[i - 1]
        for i, j in zip(range(n + 1, 2 * n + 1), range(1, 2 * n, 2)):
            A[i, j] = -1.0
            A[i, j - 1] = 1.0
            A[i, j + 1] = -1.0

        B = np.zeros((2 * n + 1, 1), dtype=complex)
        B[0] = -To

        X = linearsystem.GaussElimin(A, B, pivot_on=True, dtype=complex).solve()

        qr = X[-1]
        qr_ampl = abs(qr)
        qr_phi = cmath.phase(qr)

        r_mr = 0.5 * self.r + self._r_conv_in
        r_om = 0.5 * self.r + self._r_conv_out

        qo_ampl = (abs(To) - r_mr * qr_ampl) / r_om
        qo_phi = np.arctan2(
            np.sin(np.pi / 2.0 + qr_phi) + qr_ampl * np.sin(qr_phi),
            np.cos(np.pi / 2.0 + qr_phi) + qr_ampl * np.cos(qr_phi)
        )
        qm_ampl = np.sqrt(
            (qo_ampl * np.cos(qo_phi) - qr_ampl * np.cos(qr_phi)) ** 2
            + (qo_ampl * np.sin(qo_phi) - qr_ampl * np.sin(qr_phi)) ** 2
        )
        self._ca_eff = qm_ampl / (w * r_mr * qr_ampl)

    @property
    def ca_eff(self):
        """Specific thermal effective capacity of building part [J/(m^2.K)]"""
        return self._ca_eff


class Space:
    def __init__(self):
        self.building_parts = None                  # a space is surrounded by building parts
        self._T_in = 0.0                            # inside temperature [°C]
        self._T_out = 0.0                           # outside temperature = temperature of building environment [°C]
        self._Q_tr = 0.0                            # transmission heat loss [W]
        self._R_tr = 0.0                            # global transmission resistance [K/W]
        self._C_ra = 0.0                            # room air thermal capacity [J/K]
        self._C_bm_stat = 0.0                       # building mass static thermal capacity [J/K]
        self._C_bm_eff = 0.0                        # building mass effective thermal capacity [J/K]
        self._dim = {'l': 0.0, 'w': 0.0, 'h': 0.0}  # room dimensions [m]

    def set_temperatures(self, T_inside, T_outside):
        """Set space inside and outside temperature [°C]"""
        self._T_in = T_inside
        self._T_out = T_outside

    def set_dimensions(self, length, width, height):
        self._dim['l'] = length
        self._dim['w'] = width
        self._dim['h'] = height

    def set_building_parts(self, *building_parts):
        self.building_parts = building_parts

    def Q_tr(self):
        """Calculate transmission heat loss of space [W]"""
        self._Q_tr = 0.0  # transmission heat loss of space [W]
        for bp in self.building_parts:
            A_avg = (bp.A_in + bp.A_out) / 2.0
            self._Q_tr += bp.u * A_avg * (bp.T_in - bp.T_out)

    def R_tr(self):
        """Calculate global transmission heat resistance of space [K/W]"""
        self._R_tr = (self._T_in - self._T_out) / self._Q_tr

    def C_ra(self):
        """Calculate room air thermal capacity [J/K]"""
        V = self._dim['l'] * self._dim['w'] * self._dim['h']
        rho_air = 1.205  # [kg/m^3]
        c_air = 1005.0   # [J/(kg.K)]
        self._C_ra = rho_air * c_air * V

    def C_bm_stat(self):
        """Calculate building mass static thermal capacity [J/K]"""
        self._C_bm_stat = 0.0
        for bp in self.building_parts:
            A_avg = (bp.A_in + bp.A_out) / 2.0
            self._C_bm_stat += bp.ca * A_avg

    def C_bm_eff(self):
        """Calculate building mass effective thermal capacity [J/K]"""
        self._C_bm_eff = 0.0
        for bp in self.building_parts:
            A_avg = (bp.A_in + bp.A_out) / 2.0
            self._C_bm_eff += bp.ca_eff * A_avg

    def calculate(self):
        self.Q_tr()
        self.R_tr()
        self.C_ra()
        self.C_bm_stat()
        self.C_bm_eff()

    def get_results(self):
        return {
            'Q_tr': self._Q_tr,
            'R_tr': self._R_tr,
            'C_ra': self._C_ra,
            'C_bm_stat': self._C_bm_stat,
            'C_bm_eff': self._C_bm_eff
        }
